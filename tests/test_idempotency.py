"""
tests/test_idempotency.py
--------------------------
Verifies that firing the same failed-payment "webhook" twice doesn't
double-log — and therefore doesn't double-count recovered revenue.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Point the audit DB at a throwaway file so this test can't
    collide with a real outputs/audit_trail.db or with other tests."""
    import execution.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test_audit.db"))
    monkeypatch.setattr(db_module, "_local", type(db_module._local)())
    yield db_module


def _sample():
    txn = {
        "txn_id": "txn_dupe_1",
        "amount": 500,
        "bank": "HDFC",
        "hour": 9,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 2,
    }
    ml = {"predicted_reason": "network_error", "confidence": 0.8, "retry_success_prob": 0.7}
    decision = {
        "action": "RETRY_NOW",
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": "test",
        "source": "hard_guardrail",
    }
    return txn, ml, decision


def test_duplicate_webhook_does_not_double_log(isolated_db):
    from execution.doer import execute_and_log

    txn, ml, decision = _sample()
    first = execute_and_log(txn, ml, decision)
    second = execute_and_log(txn, ml, decision)

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["id"] == first["id"]

    rows = isolated_db.recent_rows(limit=10)
    assert len(rows) == 1


def test_different_txn_ids_both_log(isolated_db):
    from execution.doer import execute_and_log

    txn_a, ml, decision = _sample()
    txn_b = dict(txn_a, txn_id="txn_dupe_2")

    execute_and_log(txn_a, ml, decision)
    execute_and_log(txn_b, ml, decision)

    rows = isolated_db.recent_rows(limit=10)
    assert len(rows) == 2
