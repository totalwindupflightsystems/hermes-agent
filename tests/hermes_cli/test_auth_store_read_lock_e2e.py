"""A read-only auth-store transaction must outlast a slow refresh HOLDER.

GAP-057, reproduced through the real production lock: a refresh transaction keeps the auth-store
flock across its token-endpoint POST (up to the hold ceiling, 25s at defaults), while a reader only
ever wants the store loaded. With one shared 15s budget for both, the reader raised
``Timed out waiting for auth store lock`` and a whole agent tick died — against a healthy store.

The holder here is a REAL separate process holding the production flock through
``hermes_cli.auth._auth_store_lock``, exactly as ``agent/credential_pool.py``'s single-use refresh
path does, so the contention is cross-process rather than simulated.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli import auth as auth_mod
from hermes_cli.auth_constants import AUTH_LOCK_TIMEOUT_SECONDS

# Comfortably longer than the write budget a reader used to get, so a write-budget reader is
# guaranteed to lose while a read-budget reader still has room to spare.
HOLD_SECONDS = AUTH_LOCK_TIMEOUT_SECONDS + 5.0

_FRESH_ACCESS_TOKEN = "access-token-rotated-by-the-holder"
_FRESH_REFRESH_TOKEN = "refresh-token-rotated-by-the-holder"
_STALE_ACCESS_TOKEN = "access-token-consumed-by-the-holder"
_STALE_REFRESH_TOKEN = "refresh-token-consumed-by-the-holder"

# Holder: acquire the production cross-process lock, announce it, sleep past the write budget.
_HOLDER_SCRIPT = """
import sys, time
from hermes_cli.auth import _auth_store_lock
with _auth_store_lock(timeout_seconds=10.0):
    print("READY", flush=True)
    time.sleep(float(sys.argv[1]))
"""


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _bare_pool(entry: PooledCredential) -> CredentialPool:
    """A pool carrying only what the sync path touches (mirrors the other pool unit tests)."""
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = threading.RLock()
    pool._entries = [entry]
    pool._active_leases = {}
    pool._current_id = None
    pool._max_concurrent = 2
    pool._unmatched_rotation_streak = 0
    pool.provider = entry.provider
    return pool


def _start_holder(hold_seconds: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(hold_seconds)],
        cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=os.environ.copy(),
    )
    # Readiness is announced from INSIDE the held lock — never assume an elapsed time means held.
    ready = proc.stdout.readline().strip()
    if ready != "READY":
        proc.kill()
        raise AssertionError(f"holder never took the lock: {ready!r} {proc.stderr.read()!r}")
    return proc


def _write_provider_store() -> None:
    """The store the holder is 'refreshing': tokens nothing else has seen yet."""
    auth_path = auth_mod._auth_file_path()
    auth_path.write_text(
        json.dumps({
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": _FRESH_ACCESS_TOKEN,
                        "refresh_token": _FRESH_REFRESH_TOKEN,
                    },
                },
            },
        }),
        encoding="utf-8",
    )


def test_read_only_path_blocks_out_a_slow_holder_instead_of_failing(hermes_home, caplog):
    """The switched read path waits out a holder that would have starved it, then succeeds."""
    _write_provider_store()
    entry = PooledCredential(
        provider="openai-codex", id="codex-single-use", label="codex", auth_type="oauth",
        priority=0, source="device_code", access_token=_STALE_ACCESS_TOKEN,
        refresh_token=_STALE_REFRESH_TOKEN,
    )
    pool = _bare_pool(entry)
    outcome: dict = {}

    def _write_budget_reader() -> None:
        """Control: what the reader used to get — the write budget — must still lose."""
        try:
            with auth_mod._auth_store_lock():
                outcome["control"] = "acquired"
        except BaseException as exc:  # noqa: BLE001 - the exception IS the evidence
            outcome["control"] = exc

    def _read_budget_reader() -> None:
        try:
            outcome["reader"] = pool._sync_entry_from_auth_store(entry)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertions below
            outcome["reader"] = exc

    holder = _start_holder(HOLD_SECONDS)
    try:
        control = threading.Thread(target=_write_budget_reader)
        reader = threading.Thread(target=_read_budget_reader)
        started = time.monotonic()
        with caplog.at_level(logging.DEBUG, logger="agent.credential_pool"):
            control.start()
            reader.start()
            control.join(timeout=HOLD_SECONDS + 10)
            reader.join(timeout=HOLD_SECONDS + 10)
        elapsed = time.monotonic() - started
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=10)

    assert not control.is_alive() and not reader.is_alive(), "a reader thread never finished"

    # Control arm: the pre-fix budget loses to the holder, verbatim.
    assert isinstance(outcome["control"], TimeoutError), outcome["control"]
    assert str(outcome["control"]) == "Timed out waiting for auth store lock"

    # The switched path waited out the holder and came back with the store's tokens.
    assert isinstance(outcome["reader"], PooledCredential), outcome["reader"]
    assert outcome["reader"].access_token == _FRESH_ACCESS_TOKEN
    assert outcome["reader"].refresh_token == _FRESH_REFRESH_TOKEN
    # It genuinely blocked — the holder held longer than the old (write) budget.
    assert elapsed >= AUTH_LOCK_TIMEOUT_SECONDS, f"reader did not have to wait: {elapsed:.1f}s"
    assert "Timed out waiting for auth store lock" not in caplog.text, (
        "a read-only path reported the lock timeout it must no longer hit"
    )


def test_read_lock_nested_in_a_write_transaction_does_not_ref_lock(hermes_home):
    """The pool's in-lock re-sync runs under a write transaction; it must stay reentrant."""
    _write_provider_store()
    entry = PooledCredential(
        provider="openai-codex", id="codex-single-use", label="codex", auth_type="oauth",
        priority=0, source="device_code", access_token=_FRESH_ACCESS_TOKEN,
        refresh_token=_FRESH_REFRESH_TOKEN,
    )
    pool = _bare_pool(entry)

    started = time.monotonic()
    with auth_mod._auth_store_lock():
        synced = pool._sync_entry_from_auth_store(entry)
    elapsed = time.monotonic() - started

    # Same tokens on both sides: nothing to adopt, so this is the read path only.
    assert synced is entry
    assert elapsed < AUTH_LOCK_TIMEOUT_SECONDS, (
        f"the nested read re-flocked a lock this thread already held ({elapsed:.1f}s)"
    )
