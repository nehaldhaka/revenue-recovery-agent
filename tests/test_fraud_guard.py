"""
tests/test_fraud_guard.py
----------------------------
Covers decision/fraud_guard.py's card-testing detector. Uses a fresh
temp-file SQLite DB per test (via the `fresh_db` fixture) so tests
never touch outputs/audit_trail.db and never leak state between runs.
"""
import itertools
import threading
from datetime import datetime, timezone

import pytest

from execution import db
from decision import fraud_guard


_counter = itertools.count()


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Points execution.db at a throwaway SQLite file for this test only,
    and clears its thread-local connection cache so get_conn() opens a
    brand-new connection against that file instead of reusing whatever
    connection was cached from a previous test."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_audit.db"))
    monkeypatch.setattr(db, "_local", threading.local())
    yield db


def _insert_row(fresh_db, bank, amount, reason="risk_block", minutes_ago=0):
    now = datetime.now(timezone.utc).isoformat()
    fresh_db.insert_row({
        "idempotency_key": f"key_{bank}_{amount}_{minutes_ago}_{next(_counter)}",
        "logged_at": now,
        "txn_id": f"txn_{bank}_{amount}",
        "amount": amount,
        "bank": bank,
        "hour": 12,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 5,
        "predicted_reason": reason,
        "confidence": 0.5,
        "retry_success_prob": 0.5,
        "action": "ESCALATE",
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": "",
        "source": "test",
        "execution_note": "",
        "duplicate_of": None,
    })


class TestFraudGuard:
    def test_no_trip_without_bank(self, fresh_db):
        assert fraud_guard.check(None, 50) is None

    def test_no_trip_without_amount(self, fresh_db):
        assert fraud_guard.check("HDFC", None) is None

    def test_no_trip_when_amount_too_high(self, fresh_db):
        # A single ₹5000 transaction is nowhere near the card-testing
        # amount range and should never even trigger the DB lookup path.
        assert fraud_guard.check("HDFC", 5000) is None

    def test_no_trip_with_too_few_samples(self, fresh_db):
        # Below MIN_ATTEMPTS (6), even a tight low-value cluster shouldn't trip.
        for i in range(3):
            _insert_row(fresh_db, "HDFC", amount=10 + i)
        assert fraud_guard.check("HDFC", 12) is None

    def test_no_trip_when_amounts_vary_widely(self, fresh_db):
        # Same low mean, but spread out enough that stdev/mean exceeds
        # AMOUNT_TIGHT_BAND (0.9) — this is what a genuine mix of small
        # organic failures looks like, not an attack. (Verified: this
        # distribution's stdev/mean ~= 1.13.)
        amounts = [1, 99, 1, 99, 1, 99, 1]
        for amt in amounts:
            _insert_row(fresh_db, "HDFC", amount=amt)
        result = fraud_guard.check("HDFC", 20)
        assert result is None

    def test_trips_on_tight_low_value_cluster(self, fresh_db):
        # Classic card-testing fingerprint: many attempts, all low value,
        # all clustered tightly around the same amount.
        for amt in [10, 11, 9, 10, 12, 10, 11]:
            _insert_row(fresh_db, "YesBank", amount=amt)
        result = fraud_guard.check("YesBank", 10)
        assert result is not None
        assert result["action"] == "ESCALATE_TO_HUMAN"
        assert result["source"] == "fraud_guard"
        assert "card-testing" in result["reasoning"].lower()

    def test_does_not_trip_for_a_different_bank(self, fresh_db):
        # The cluster is entirely on YesBank; checking HDFC (which has no
        # history at all) must not be affected by another bank's pattern.
        for amt in [10, 11, 9, 10, 12, 10, 11]:
            _insert_row(fresh_db, "YesBank", amount=amt)
        assert fraud_guard.check("HDFC", 10) is None