"""Who stopped a background process, and what accounting was in force when it died.

A reader thread that sees EOF can only report a returncode. When that returncode is
NEGATIVE the process did not exit — a signal killed it — yet the registry used to record
``exit_code: -9, completion_reason: "exited", termination_source: ""``, which reads to every
consumer (and to the operator reading the receipt) exactly like a process that finished on
its own. Measured on a fleet worker (DF-CRIER-463): SIGKILL at 9m59s wall, 1.6G peak, a
43-byte log, and no harness log line anywhere naming a killer.

This module turns a signal death into durable, checkable facts:

* ``termination_source`` — the killer, taken only from evidence that outlives the process,
  in this order: the registry's own kill record (recorded BEFORE the signal is sent, so a
  kill that races the reader cannot be mis-attributed), the worker cgroup's ``oom_kill``
  counter, the systemd unit's ``Result``. A signal death with none of those is
  ``unknown_no_supervisor_stop`` — never an empty string, and never a guess at one of the
  usual suspects.
* ``termination_detail`` — one sentence naming the signal, the elapsed time and every
  surface that was checked with its answer, so the next occurrence is diagnosable from the
  receipt alone.
* ``termination_accounting`` — the caps in force at kill time, each with whether it applied,
  so a suspiciously round boundary (a ~600s wall for a 600s foreground cap and a 600s
  one-shot linger) is either named as a policy or shown not to be one.
* ``dirty_worktree`` — the git state a dying worker left behind, so salvage does not need a
  manual tree inspection.

Boundary, stated rather than implied: an outside SIGKILL leaves no in-process trace. Where
this module cannot prove a killer it records that it could not, which is the only claim a
receipt may make. Attribution is evidence about a surface that outlived the process; it is
never an inference from timing.
"""

import logging
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession

logger = logging.getLogger("tools.process_registry")

#: Recorded for a signal death whose killer no surviving surface names. A sentence, not a
#: shrug: the row that produced this module asked for exactly this token instead of "".
UNKNOWN_STOP = "unknown_no_supervisor_stop"

#: Sources that are evidence of a stop even when the exit code is positive (a backend that
#: vanished, a launch that never produced a PID).
_NON_SIGNAL_STOPS = {"backend_lost", "failed_start"}

#: `git status` on a cold worktree with a huge index is the only unbounded part of finalize.
_WORKTREE_PROBE_TIMEOUT_SECONDS = 3.0
#: Exit evidence only has to be readable for one sentence; a slow D-Bus must not stall a finish.
_SYSTEMD_QUERY_TIMEOUT_SECONDS = 2.0
_WORKTREE_SAMPLE_FILES = 5


def is_signal_exit(exit_code) -> bool:
    """True when a returncode is a signal number: the process was killed, not exited."""
    return isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code < 0


def signal_name(exit_code) -> str:
    """``SIGKILL`` for -9; empty for a real exit; ``signal 34`` for a number this host
    cannot name (a real-time signal has no ``Signals`` member everywhere)."""
    if not is_signal_exit(exit_code):
        return ""
    import signal as _signal

    try:
        return _signal.Signals(-exit_code).name
    except (ValueError, AttributeError):
        return f"signal {-exit_code}"


def reported_signal(exit_code) -> str:
    """Signal name from either convention: ``Popen.returncode`` (negative) or a shell's
    ``128+N`` (``$?`` after a signal death). Empty when the code is a real exit."""
    try:
        code = int(exit_code)
    except (TypeError, ValueError):
        return ""
    if code < 0:
        return signal_name(code)
    return signal_name(-(code - 128)) if code > 128 else ""


def cgroup_path_for_pid(pid: Optional[int]) -> str:
    """The cgroup-v2 directory of a live PID, or "" when unavailable.

    Recorded at spawn because it is the ONLY place a cgroup OOM leaves evidence, and the
    directory is gone once the process (and its transient scope) is collected.
    """
    if not pid:
        return ""
    try:
        lines = Path(f"/proc/{int(pid)}/cgroup").read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError, TypeError):
        return ""
    for line in lines:
        if line.startswith("0::"):
            relative = line.partition("::")[2].strip()
            return f"/sys/fs/cgroup{relative}" if relative.startswith("/") else ""
    return ""  # cgroup v1 or no unified hierarchy: no memory.events to read


def _read_int(path: Path) -> Optional[int]:
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    return int(text) if text.isdigit() else None


def _read_key_values(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        key, _, value = line.partition(" ")
        if key and value.strip().isdigit():
            values[key] = int(value.strip())
    return values


def cgroup_evidence(session: "ProcessSession") -> dict:
    """The worker cgroup's own OOM counters, if the directory still exists.

    ``oom_kill`` is the kernel's count of processes in THIS cgroup killed for hitting its
    ``memory.max`` — the one signature a cgroup OOM cannot fake. An unreadable directory is
    reported as such (``--collect`` removes a transient scope when its last process exits),
    never as "no OOM".
    """
    evidence = {
        "path": getattr(session, "cgroup_path", "") or "",
        "readable": False,
        "oom_kill": None,
        "memory_peak_bytes": None,
        "memory_max_bytes": None,
    }
    if not evidence["path"]:
        return evidence
    directory = Path(evidence["path"])
    with suppress(OSError, ValueError):
        evidence["oom_kill"] = _read_key_values(directory / "memory.events").get("oom_kill")
        evidence["readable"] = True
    with suppress(OSError, ValueError):
        evidence["memory_peak_bytes"] = _read_int(directory / "memory.peak")
    with suppress(OSError, ValueError):
        evidence["memory_max_bytes"] = _read_int(directory / "memory.max")
    return evidence


def systemd_evidence(session: "ProcessSession") -> dict:
    """The transient scope's own verdict (``Result``/``ExecMainStatus``), when queryable.

    A scope spawned with ``--collect`` usually disappears with its last process, so
    ``queried=False`` is the common answer and is reported as "not queryable", not as a
    clean bill of health.
    """
    unit = getattr(session, "systemd_unit", "") or ""
    evidence = {"unit": unit, "queried": False, "result": "", "exec_main_status": None,
                "active_state": "", "error": ""}
    if not unit:
        return evidence
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        evidence["error"] = "systemctl not on PATH"
        return evidence
    try:
        from tools.process_registry import systemd_user_bus_env

        result = subprocess.run(
            [binary, "--user", "show", unit, "-p", "Result", "-p", "ExecMainStatus", "-p", "ActiveState"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            stdin=subprocess.DEVNULL, timeout=_SYSTEMD_QUERY_TIMEOUT_SECONDS,
            env=systemd_user_bus_env())
    except (OSError, subprocess.SubprocessError) as exc:
        evidence["error"] = str(exc)
        return evidence
    if result.returncode != 0:
        evidence["error"] = (result.stderr or "").strip()[:200] or f"systemctl rc={result.returncode}"
        return evidence
    fields = {}
    for line in (result.stdout or "").splitlines():
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip()
    if not fields:
        evidence["error"] = "systemctl returned no properties"
        return evidence
    evidence.update(
        queried=True,
        result=fields.get("Result", ""),
        active_state=fields.get("ActiveState", ""),
        exec_main_status=(
            int(fields["ExecMainStatus"]) if fields.get("ExecMainStatus", "").lstrip("-").isdigit() else None),
    )
    return evidence


def _foreground_cap_seconds() -> Optional[int]:
    """The terminal tool's foreground cap, read without importing it (the caller already
    imported it, so this is a dict lookup; a lazy import here could pull the whole tool
    stack into a finish path that must stay cheap)."""
    module = sys.modules.get("tools.terminal_tool")
    value = getattr(module, "FOREGROUND_MAX_TIMEOUT", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _oneshot_wait_seconds() -> Optional[float]:
    try:
        from tools.process_registry import ProcessRegistry

        return float(ProcessRegistry._oneshot_completion_wait_seconds())
    except (ImportError, AttributeError, TypeError, ValueError):
        # Reporting no cap is correct here: the receipt says "not applicable", never a
        # fabricated 0 that would read like a policy.
        return None


def _cgroup_memory_max_bytes() -> Optional[int]:
    try:
        from tools.process_registry import _worker_memory_max_bytes

        return int(_worker_memory_max_bytes())
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def accounting_snapshot(session: "ProcessSession") -> dict:
    """Every cap the harness could have applied to this process, each with ``applies``.

    Spelled out rather than summarised because the question this answers is "was a ~600s
    boundary a policy?" — and the harness has two 600s constants that do NOT signal a
    background process. Naming them, with ``applies: false``, is what makes the absence of a
    cap a measurement instead of an assertion.
    """
    started = getattr(session, "started_at", 0.0) or 0.0
    scoped = bool(getattr(session, "systemd_unit", ""))
    return {
        "elapsed_seconds": round(time.time() - started, 3) if started else None,
        "caps": {
            "wall_clock": {
                "value": None, "unit": "seconds", "applies": False,
                "note": "the harness sets no runtime cap on background processes",
            },
            "terminal_foreground_timeout": {
                "value": _foreground_cap_seconds(), "unit": "seconds", "applies": False,
                "note": "foreground calls only; it never signals a tracked background session",
            },
            "oneshot_completion_wait": {
                "value": _oneshot_wait_seconds(), "unit": "seconds", "applies": False,
                "note": "a bounded linger at one-shot exit; it waits, it does not signal",
            },
            "cgroup_memory_max": {
                "value": _cgroup_memory_max_bytes() if scoped else None, "unit": "bytes",
                "applies": scoped,
                "note": "per-worker systemd scope limit" if scoped
                        else "no isolated worker cgroup for this spawn",
            },
        },
        "pid_scope": getattr(session, "pid_scope", ""),
        "systemd_unit": getattr(session, "systemd_unit", ""),
    }


def _elapsed_phrase(accounting: dict) -> str:
    elapsed = accounting.get("elapsed_seconds")
    return f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "an unknown elapsed time"


def _summarise_checks(cgroup: dict, systemd: dict) -> str:
    """One clause per surface that was consulted, with its answer — the sentence a reader
    needs when the token says "unknown"."""
    if not cgroup.get("path"):
        cgroup_clause = "no worker cgroup was recorded at spawn"
    elif cgroup.get("readable"):
        cgroup_clause = (
            f"cgroup oom_kill={cgroup.get('oom_kill')} "
            f"peak={cgroup.get('memory_peak_bytes')} max={cgroup.get('memory_max_bytes')}")
    else:
        cgroup_clause = f"worker cgroup {cgroup.get('path')} was gone at exit (scope collected)"
    if not systemd.get("unit"):
        systemd_clause = "no systemd scope"
    elif systemd.get("queried"):
        systemd_clause = (
            f"unit {systemd['unit']} Result={systemd.get('result') or '?'} "
            f"ExecMainStatus={systemd.get('exec_main_status')}")
    else:
        systemd_clause = (
            f"unit {systemd['unit']} not queryable ({systemd.get('error') or 'unit collected'})")
    return f"cgroup: {cgroup_clause}; systemd: {systemd_clause}"


def attribute(session: "ProcessSession", exit_code, reason: str, requested_source: str,
              elapsed: Optional[float] = None) -> tuple:
    """``(termination_source, termination_detail)`` for one finished process.

    ``requested_source`` is what the registry recorded BEFORE it signalled anything (or what
    a caller already knew: a vanished backend, a launch that never produced a PID). It wins
    over every post-mortem surface, because a scope teardown issued by this process would
    otherwise look like a systemd stop it never asked for.
    """
    if elapsed is None:
        started = getattr(session, "started_at", 0.0) or 0.0
        elapsed = round(time.time() - started, 3) if started else None
    elapsed_text = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "an unknown elapsed time"

    if requested_source:
        return requested_source, f"stopped on request ({requested_source}) after {elapsed_text}"
    if reason in _NON_SIGNAL_STOPS:
        return reason, f"recorded {reason} after {elapsed_text}"

    if not is_signal_exit(exit_code):
        code = "no return code" if exit_code is None else f"exit code {exit_code}"
        return "", f"exited with {code} after {elapsed_text}; no signal was delivered"

    killed_by = signal_name(exit_code)
    cgroup = cgroup_evidence(session)
    systemd = systemd_evidence(session)
    checks = _summarise_checks(cgroup, systemd)

    if (cgroup.get("oom_kill") or 0) > 0:
        return "cgroup_oom", (
            f"{killed_by} after {elapsed_text}: the worker cgroup recorded oom_kill="
            f"{cgroup['oom_kill']} (peak {cgroup.get('memory_peak_bytes')}B of "
            f"{cgroup.get('memory_max_bytes')}B); {checks}")
    if systemd.get("queried") and systemd.get("result") == "signal":
        return "systemd", (
            f"{killed_by} after {elapsed_text}: systemd reports Result=signal "
            f"ExecMainStatus={systemd.get('exec_main_status')} for {systemd['unit']}, and no "
            f"registry kill was recorded; {checks}")
    return UNKNOWN_STOP, (
        f"{killed_by} after {elapsed_text} with no supervisor recorded a stop "
        f"(no registry kill request; {checks})")


def worktree_report(cwd: Optional[str]) -> dict:
    """Git state of a dying process's cwd — the salvage list, not a cleanliness verdict.

    Empty dict when the cwd is not a git work tree or is clean: a receipt is not a place to
    report "clean" on every ordinary exit.
    """
    if not cwd or not os.path.isdir(cwd):
        return {}
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain=v1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            stdin=subprocess.DEVNULL, timeout=_WORKTREE_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return {}
    # Porcelain v1 is two fixed status columns then a space: X = index, Y = worktree.
    staged = sum(1 for line in lines if len(line) > 1 and line[0] not in " ?")
    unstaged = sum(1 for line in lines if len(line) > 1 and line[1] not in " ?")
    untracked = sum(1 for line in lines if line.startswith("??"))
    return {
        "dirty": True,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "changed": len(lines),
        "sample": [line[3:] if len(line) > 3 else line for line in lines[:_WORKTREE_SAMPLE_FILES]],
        "truncated": len(lines) > _WORKTREE_SAMPLE_FILES,
    }


def _salvage_matters(session: "ProcessSession") -> bool:
    """True when this finish could have lost work: a signal death, a lost backend, or a
    non-zero exit. A clean exit 0 is not a salvage event."""
    exit_code = getattr(session, "exit_code", None)
    reason = getattr(session, "completion_reason", "exited")
    if is_signal_exit(exit_code) or reason in _NON_SIGNAL_STOPS:
        return True
    return isinstance(exit_code, int) and exit_code != 0


def dirty_worktree_line(dirty_worktree: dict) -> str:
    """Human sentence for the completion notice; "" when there is nothing to salvage."""
    if not dirty_worktree.get("dirty"):
        return ""
    return (
        f"Uncommitted work in the worktree: {dirty_worktree.get('staged', 0)} staged, "
        f"{dirty_worktree.get('unstaged', 0)} unstaged, {dirty_worktree.get('untracked', 0)} "
        f"untracked file(s) — salvage before the worktree is reaped.")


def finalize(session: "ProcessSession") -> None:
    """Fill attribution, accounting and salvage evidence once, before the receipt is written.

    Called from the single finish path (``ProcessRegistry._move_to_finished``) so every way a
    process can end — reader EOF, explicit kill, sandbox poller, reconciled orphan — is
    covered by one implementation. Idempotent: a duplicate finisher must not re-probe git.
    """
    if getattr(session, "_finalized", False):
        return
    session._finalized = True

    exit_code = getattr(session, "exit_code", None)
    requested = getattr(session, "termination_source", "") or getattr(session, "kill_requested_source", "")
    accounting = accounting_snapshot(session)
    source, detail = attribute(
        session, exit_code, getattr(session, "completion_reason", "exited"), requested,
        elapsed=accounting.get("elapsed_seconds"))

    # A negative returncode is a kill, not an exit. Recording it as "exited" is exactly what
    # made a SIGKILLed worker indistinguishable from a finished one in every consumer.
    # Only a plain-exit reason qualifies: "lost"/"failed_start" carry a -1 SENTINEL (no signal
    # arrived at all), and rewriting those to "killed" would invent a killer.
    if is_signal_exit(exit_code) and getattr(session, "completion_reason", "") == "exited":
        session.completion_reason = "killed"

    if source:
        session.termination_source = source
    session.termination_detail = detail
    session.termination_accounting = accounting
    # Only a HOST cwd is a path this process can read: a sandbox session's cwd lives inside
    # that backend, and probing the same string locally would report a stranger's tree.
    if _salvage_matters(session) and getattr(session, "pid_scope", "host") == "host":
        session.dirty_worktree = worktree_report(getattr(session, "cwd", "") or "")

    caps = accounting.get("caps", {})
    logger.info(
        "process %s ended after %s: source=%s reason=%s exit=%s | accounting: wall_cap=%s "
        "(applies=%s), foreground_cap=%ss (applies=%s), memory_max=%sB (applies=%s), unit=%s | %s",
        getattr(session, "id", "?"), _elapsed_phrase(accounting),
        session.termination_source or "none-recorded", session.completion_reason, exit_code,
        caps.get("wall_clock", {}).get("value"), caps.get("wall_clock", {}).get("applies"),
        caps.get("terminal_foreground_timeout", {}).get("value"),
        caps.get("terminal_foreground_timeout", {}).get("applies"),
        caps.get("cgroup_memory_max", {}).get("value"), caps.get("cgroup_memory_max", {}).get("applies"),
        accounting.get("systemd_unit") or "none", detail)
    salvage = dirty_worktree_line(session.dirty_worktree)
    if salvage:
        logger.warning("process %s died with uncommitted work in %s: %s",
                       getattr(session, "id", "?"), getattr(session, "cwd", "?"), salvage)
