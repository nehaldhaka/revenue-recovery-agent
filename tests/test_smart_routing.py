"""
tests/test_smart_routing.py
------------------------------
Covers decision/smart_routing.py's suggest_route(). Uses the same
temp-DB fixture pattern as test_fraud_guard.py.
"""
import itertools
import threading
from datetime import datetime, timezone

import pytest

from execution import db
from decision import smart_routing


_counter = itertools.count()


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_audit.db"))
    monkeypatch.setattr(db, "_local", threading.local())
    yield db


def _insert_row(fresh_db, bank, reason, outcome=None, amount=2000):
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "idempotency_key": f"key_{bank}_{reason}_{outcome}_{next(_counter)}",
        "logged_at": now,
        "txn_id": f"txn_{bank}_{reason}_{next(_counter)}",
        "amount": amount,
        "bank": bank,
        "hour": 12,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 5,
        "predicted_reason": reason,
        "confidence": 0.7,
        "retry_success_prob": 0.6,
        "action": "RETRY_NOW",
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": "",
        "source": "test",
        "execution_note": "",
        "duplicate_of": None,
    }
    saved = fresh_db.insert_row(row)
    if outcome is not None:
        fresh_db.record_outcome(row["txn_id"], recovered=(outcome == "recovered"))
    return saved


class TestSmartRouting:
    def test_no_recommendation_with_no_history(self, fresh_db):
        result = smart_routing.suggest_route("YesBank", "bank_timeout")
        assert result["recommended_bank"] is None
        assert result["candidates"] == {}

    def test_no_recommendation_below_min_samples(self, fresh_db):
        # Only 2 outcomes for HDFC — below MIN_ROUTE_SAMPLES (3).
        _insert_row(fresh_db, "HDFC", "bank_timeout", "recovered")
        _insert_row(fresh_db, "HDFC", "bank_timeout", "recovered")
        result = smart_routing.suggest_route("YesBank", "bank_timeout")
        assert result["recommended_bank"] is None

    def test_recommends_bank_with_best_recovery_rate(self, fresh_db):
        # HDFC: 3/4 recovered (75%). ICICI: 2/4 recovered (50%).
        # Failing bank is YesBank, so it should never be a candidate.
        for outcome in ["recovered", "recovered", "recovered", "failed"]:
            _insert_row(fresh_db, "HDFC", "bank_timeout", outcome)
        for outcome in ["recovered", "recovered", "failed", "failed"]:
            _insert_row(fresh_db, "ICICI", "bank_timeout", outcome)

        result = smart_routing.suggest_route("YesBank", "bank_timeout")
        assert result["recommended_bank"] == "HDFC"
        assert result["recovery_rate"] == 0.75
        assert "YesBank" not in result["candidates"]

    def test_ignores_rows_with_different_reason(self, fresh_db):
        # HDFC has plenty of history, but all for a different failure
        # reason — it should not count toward the bank_timeout comparison.
        for outcome in ["recovered", "recovered", "recovered", "recovered"]:
            _insert_row(fresh_db, "HDFC", "insufficient_funds", outcome)
        result = smart_routing.suggest_route("YesBank", "bank_timeout")
        assert result["recommended_bank"] is None

    def test_ignores_rows_without_a_recorded_outcome(self, fresh_db):
        # Escalated/pending cases with no outcome yet shouldn't count as
        # evidence either way.
        for _ in range(5):
            _insert_row(fresh_db, "HDFC", "bank_timeout", outcome=None)
        result = smart_routing.suggest_route("YesBank", "bank_timeout")
        assert result["recommended_bank"] is None