"""A killed worker must say who killed it, what was in force, and what it left behind.

DF-CRIER-463 measured the defect on a live fleet worker: SIGKILLed at 9m59s, recorded as
``exit_code: -9, completion_reason: "exited", termination_source: ""`` — indistinguishable
from a process that finished on its own. These are behaviour contracts, driven through the
real registry with real processes (a real ``os.kill`` for the kill cases), not bare-function
calls against hand-built state.
"""

import os
import signal
import subprocess
import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_attribution import (
    UNKNOWN_STOP,
    accounting_snapshot,
    attribute,
    cgroup_path_for_pid,
    dirty_worktree_line,
    is_signal_exit,
    reported_signal,
    signal_name,
    worktree_report,
)


def _session(**kwargs) -> ProcessSession:
    kwargs.setdefault("id", "proc_attr_test")
    kwargs.setdefault("command", "sleep 30")
    kwargs.setdefault("started_at", time.time())
    return ProcessSession(**kwargs)


def _git(cwd, *args) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"})


def _dirty_repo(tmp_path):
    repo = tmp_path / "worktree"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "seed")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")   # unstaged
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")     # untracked
    return repo


def _spawn_and_kill(registry: ProcessRegistry, cwd, sig=signal.SIGKILL) -> ProcessSession:
    """Spawn a real local process, SIGKILL it from outside, and wait for the registry's
    reader thread to finish the session — the measured shape of the incident.

    One signal only: re-sending it races the live-system guard's liveness probe (and the real
    world has no such retry). The wait is on the session's completion event, never on a delay.
    """
    session = registry.spawn_local("sleep 30", cwd=str(cwd))
    assert session.pid
    os.kill(session.pid, sig)
    assert session._completion_event.wait(timeout=10), "reader thread never finished the session"
    return session


def test_signal_death_records_a_source_and_the_signal(tmp_path):
    registry = ProcessRegistry()
    session = _spawn_and_kill(registry, tmp_path)

    assert session.exit_code == -signal.SIGKILL
    assert session.completion_reason == "killed", "a -9 is a kill, not an exit"
    # Nothing in this process asked for the kill and no cgroup/unit evidence exists here,
    # so the honest answer is the explicit token — never "".
    assert session.termination_source == UNKNOWN_STOP
    assert "SIGKILL" in session.termination_detail
    assert "no supervisor recorded a stop" in session.termination_detail


def test_the_elapsed_time_and_the_accounting_models_are_recorded(tmp_path):
    registry = ProcessRegistry()
    session = _spawn_and_kill(registry, tmp_path)
    accounting = session.termination_accounting

    assert isinstance(accounting.get("elapsed_seconds"), (int, float))
    assert accounting["elapsed_seconds"] >= 0
    caps = accounting["caps"]
    # The wall-clock question the row asks: is there a ~600s policy? Every candidate model is
    # named, and none of them applies to a background process.
    assert caps["wall_clock"]["applies"] is False
    assert caps["terminal_foreground_timeout"]["applies"] is False
    assert caps["oneshot_completion_wait"]["applies"] is False
    assert "wall_clock" in caps


def test_a_signalled_worker_with_a_dirty_worktree_is_flagged(tmp_path):
    registry = ProcessRegistry()
    repo = _dirty_repo(tmp_path)
    session = _spawn_and_kill(registry, repo)

    dirty = session.dirty_worktree
    assert dirty.get("dirty") is True
    assert dirty["staged"] == 1
    assert dirty["unstaged"] == 1
    assert dirty["untracked"] == 1
    assert "salvage" in dirty_worktree_line(dirty)


def test_a_registry_kill_keeps_its_own_attribution(tmp_path):
    """Our own stop must never be re-attributed by the post-mortem surfaces: the reader can
    finish the session mid-signal, and a scope teardown looks exactly like a systemd kill."""
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 30", cwd=str(tmp_path))
    try:
        result = registry.kill_process(session.id)
    finally:
        registry._running.pop(session.id, None)

    assert result["status"] == "killed"
    assert session.termination_source == "process.kill"
    assert "stopped on request" in session.termination_detail


def test_a_clean_exit_is_not_reported_as_a_salvage_event(tmp_path):
    registry = ProcessRegistry()
    repo = _dirty_repo(tmp_path)
    session = registry.spawn_local("true", cwd=str(repo))
    assert session._completion_event.wait(timeout=10)

    assert session.exit_code == 0
    assert session.completion_reason == "exited"
    assert session.termination_source == ""
    assert session.dirty_worktree == {}, "an exit 0 has no work to salvage"
    assert "no signal was delivered" in session.termination_detail


def test_a_failed_worker_in_a_dirty_tree_is_still_flagged(tmp_path):
    registry = ProcessRegistry()
    repo = _dirty_repo(tmp_path)
    session = registry.spawn_local("exit 3", cwd=str(repo))
    assert session._completion_event.wait(timeout=10)

    assert session.exit_code == 3
    assert session.dirty_worktree.get("dirty") is True


def test_cgroup_oom_evidence_is_named_when_the_counter_proves_it(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup" / "hermes-worker-proc_x.scope"
    cgroup.mkdir(parents=True)
    (cgroup / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 2\noom_kill 2\n", encoding="utf-8")
    (cgroup / "memory.peak").write_text("1717986918\n", encoding="utf-8")
    (cgroup / "memory.max").write_text("4294967296\n", encoding="utf-8")
    session = _session(cgroup_path=str(cgroup), systemd_unit="hermes-worker-proc_x")

    source, detail = attribute(session, -signal.SIGKILL, "killed", "")

    assert source == "cgroup_oom"
    assert "oom_kill=2" in detail
    assert "4294967296" in detail


def test_an_unrecorded_stop_names_every_surface_it_checked(tmp_path):
    """The receipt has to be diagnosable without a second investigation: the token says
    "unknown" and the sentence says what was consulted and what it answered."""
    gone = tmp_path / "gone"
    session = _session(cgroup_path=str(gone))
    source, detail = attribute(session, -signal.SIGKILL, "killed", "")
    assert source == UNKNOWN_STOP
    assert "gone at exit" in detail, detail

    session = _session()
    _, detail_no_scope = attribute(session, -signal.SIGKILL, "killed", "")
    assert "no worker cgroup was recorded at spawn" in detail_no_scope
    assert "no systemd scope" in detail_no_scope


def test_signal_helpers_handle_both_conventions():
    assert is_signal_exit(-9) and not is_signal_exit(0) and not is_signal_exit(None)
    assert signal_name(-9) == "SIGKILL"
    assert signal_name(0) == ""
    assert reported_signal(-15) == "SIGTERM"
    assert reported_signal(137) == "SIGKILL", "a shell's 128+N is a signal death too"
    assert reported_signal(0) == ""


def test_accounting_snapshot_always_names_the_elapsed_time():
    snapshot = accounting_snapshot(_session())
    assert snapshot["elapsed_seconds"] is not None
    assert snapshot["caps"]["wall_clock"]["value"] is None
    assert snapshot["caps"]["wall_clock"]["applies"] is False


def test_cgroup_path_of_this_process_is_resolvable():
    """The spawn-time capture is only useful if the path is real: prove it on this PID."""
    path = cgroup_path_for_pid(os.getpid())
    if not path:
        pytest.skip("no cgroup-v2 unified hierarchy on this host")
    assert os.path.isdir(path)


def test_worktree_report_is_empty_for_a_non_repository(tmp_path):
    assert worktree_report(str(tmp_path)) == {}
    assert worktree_report("") == {}
    assert worktree_report("/does/not/exist") == {}


def test_the_retained_receipt_carries_the_attribution(tmp_path):
    """The incident's artefact was ``logs/process-results/<id>.json`` — the receipt a
    post-mortem reads. If the fields stop reaching that file, the same empty string returns."""
    import json

    from hermes_constants import get_hermes_home

    registry = ProcessRegistry()
    repo = _dirty_repo(tmp_path)
    session = _spawn_and_kill(registry, repo)

    path = get_hermes_home() / "logs" / "process-results" / f"{session.id}.json"
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    assert receipt["exit_code"] == -signal.SIGKILL
    assert receipt["completion_reason"] == "killed"
    assert receipt["termination_source"] == UNKNOWN_STOP
    assert "SIGKILL" in receipt["termination_detail"]
    assert receipt["termination_accounting"]["caps"]["wall_clock"]["applies"] is False
    assert receipt["dirty_worktree"] == {
        "dirty": True, "staged": 1, "unstaged": 1, "untracked": 1, "changed": 3,
        "sample": ["staged.txt", "tracked.txt", "untracked.txt"], "truncated": False,
    }


def test_every_receipt_field_has_a_default_so_older_receipts_still_load():
    """A receipt written before a field existed must load with its default instead of being
    skipped as unreadable: a receipt that silently disappears is worse than a sparse one."""
    from tools.process_registry_results import _RESULT_FIELDS, _result_field_defaults

    defaults = _result_field_defaults()
    # `id`/`command` are required (no default) and every producer has always written them.
    optional = set(_RESULT_FIELDS) - {"id", "command"}
    assert optional <= set(defaults), optional - set(defaults)
