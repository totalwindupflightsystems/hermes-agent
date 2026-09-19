"""The auth-store lock's READ budget must cover the worst-case HOLDER budget.

Every Hermes process serializes auth.json transactions on one cross-process flock. A refresh
transaction legitimately holds that lock across its token-endpoint POST — up to
``AUTH_LOCK_HOLD_CEILING_SECONDS`` — while a reader only ever wants the store loaded. When both
jobs shared one budget (the 15s write budget) every read that collided with a slow refresh raised
``Timed out waiting for auth store lock`` and took a whole agent tick with it, against a store that
was perfectly healthy.

These tests assert the RELATIONSHIP between the budgets (and between the holder budgets the refresh
paths actually compute at runtime and the ceiling), never a frozen value, so retuning any single
constant stays a deliberate decision instead of a silent inversion.
"""

from __future__ import annotations

import pytest

import hermes_cli.auth_constants as constants
from agent.credential_pool import CredentialPool
from hermes_cli.auth_nous import _shared_lock_timeout

# Every env override the refresh paths honour for their POST budget; unset them so the runtime
# holder budgets below are the DEFAULT worst case the ceiling is sized against.
_REFRESH_TIMEOUT_ENV_VARS = (
    "HERMES_ANTHROPIC_REFRESH_TIMEOUT_SECONDS",
    "HERMES_CODEX_REFRESH_TIMEOUT_SECONDS",
    "HERMES_XAI_REFRESH_TIMEOUT_SECONDS",
)

_SINGLE_USE_REFRESH_PROVIDERS = ("openai-codex", "xai-oauth", "anthropic")


def _bare_pool(provider: str) -> CredentialPool:
    """A pool that only carries what the holder-budget computation reads."""
    pool = CredentialPool.__new__(CredentialPool)
    pool.provider = provider
    return pool


@pytest.fixture
def default_refresh_timeouts(monkeypatch):
    for var in _REFRESH_TIMEOUT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_read_budget_outlasts_the_hold_ceiling():
    """A reader must be able to wait out any holder it can collide with."""
    assert constants.AUTH_LOCK_READ_TIMEOUT_SECONDS >= constants.AUTH_LOCK_HOLD_CEILING_SECONDS
    # The bug this prevents: equal budgets meant every colliding read lost.
    assert constants.AUTH_LOCK_READ_TIMEOUT_SECONDS > constants.AUTH_LOCK_TIMEOUT_SECONDS
    assert constants.AUTH_LOCK_HOLD_CEILING_SECONDS >= constants.AUTH_LOCK_TIMEOUT_SECONDS
    assert constants.AUTH_LOCK_READ_MARGIN_SECONDS > 0, (
        "a read needs room for the holder's write leg after its POST"
    )
    # The ceiling is the holder formula applied to the shared default POST budget.
    assert constants.AUTH_LOCK_HOLD_CEILING_SECONDS == constants.auth_lock_hold_timeout(
        constants.AUTH_REFRESH_POST_TIMEOUT_SECONDS
    )


def test_every_holder_budget_fits_under_the_hold_ceiling(default_refresh_timeouts):
    """The holder budgets the refresh paths compute must not exceed the ceiling."""
    holder_budgets = {
        "default holder formula": constants.auth_lock_hold_timeout(),
        "nous shared-store holder": _shared_lock_timeout(constants.AUTH_LOCK_TIMEOUT_SECONDS),
    }
    for provider in _SINGLE_USE_REFRESH_PROVIDERS:
        holder_budgets[f"{provider} single-use refresh"] = (
            _bare_pool(provider)._single_use_refresh_lock_timeout()
        )

    for label, budget in holder_budgets.items():
        assert budget <= constants.AUTH_LOCK_HOLD_CEILING_SECONDS, (
            f"{label} may hold the lock for {budget}s, past the "
            f"{constants.AUTH_LOCK_HOLD_CEILING_SECONDS}s ceiling"
        )
        # Holders never get less than the write budget, and readers always outlast them.
        assert budget >= constants.AUTH_LOCK_TIMEOUT_SECONDS, label
        assert constants.AUTH_LOCK_READ_TIMEOUT_SECONDS > budget, label


def test_configured_refresh_timeout_still_extends_the_holder_budget(default_refresh_timeouts):
    """An explicit env override widens the holder budget — the ceiling covers DEFAULTs only."""
    assert constants.auth_lock_hold_timeout(constants.AUTH_REFRESH_POST_TIMEOUT_SECONDS) == (
        constants.AUTH_LOCK_HOLD_CEILING_SECONDS
    )
    longer = constants.AUTH_REFRESH_POST_TIMEOUT_SECONDS + 30.0
    assert constants.auth_lock_hold_timeout(longer) > constants.AUTH_LOCK_HOLD_CEILING_SECONDS
