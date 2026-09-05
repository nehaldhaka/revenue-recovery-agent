"""
tests/test_circuit_breaker.py
-------------------------------
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    import execution.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test_audit.db"))
    monkeypatch.setattr(db_module, "_local", type(db_module._local)())
    yield db_module


def _log_bank_timeout(bank, i):
    from execution.doer import execute_and_log

    txn = {
        "txn_id": f"txn_{bank}_{i}",
        "amount": 500,
        "bank": bank,
        "hour": 9,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 2,
    }
    ml = {"predicted_reason": "bank_timeout", "confidence": 0.8, "retry_success_prob": 0.7}
    decision = {
        "action": "ESCALATE_TO_HUMAN",
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": "test",
        "source": "hard_guardrail",
    }
    execute_and_log(txn, ml, decision)


def test_breaker_trips_after_repeated_bank_side_failures(isolated_db):
    from decision.circuit_breaker import check, MIN_SAMPLES

    for i in range(MIN_SAMPLES):
        _log_bank_timeout("YesBank", i)

    result = check("YesBank")
    assert result is not None
    assert result["action"] == "ESCALATE_TO_HUMAN"
    assert result["source"] == "circuit_breaker"


def test_breaker_does_not_trip_with_too_few_samples(isolated_db):
    from decision.circuit_breaker import check

    # Only log 2 failures — below MIN_SAMPLES, so the breaker should
    # stay closed regardless of how bad the failure rate looks.
    _log_bank_timeout("HDFC", 1)
    _log_bank_timeout("HDFC", 2)

    assert check("HDFC") is None


def test_breaker_ignores_unrelated_bank(isolated_db):
    from decision.circuit_breaker import check, MIN_SAMPLES

    for i in range(MIN_SAMPLES):
        _log_bank_timeout("IDFC", i)

    # A bank with no history of its own shouldn't be affected by
    # another bank's failures.
    assert check("SBI") is None
