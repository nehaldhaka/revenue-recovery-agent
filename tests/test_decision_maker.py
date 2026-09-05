"""
tests/test_decision_maker.py
-----------------------------
Covers decide()'s boundary cases: the hard guardrails (which must
always win, regardless of what the ML layer or the mock/LLM would
otherwise pick) and the mock decision branches used when no
ANTHROPIC_API_KEY is set.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decision.decision_maker import (  # noqa: E402
    decide,
    MAX_RETRIES,
    LOW_CONFIDENCE_THRESHOLD,
    AUTO_ACTION_AMOUNT_CAP,
)


def base_txn(**overrides):
    txn = {
        "txn_id": "txn_test_1",
        "amount": 1000,
        "bank": "HDFC",
        "hour": 10,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 5,
    }
    txn.update(overrides)
    return txn


def base_ml(**overrides):
    ml = {
        "predicted_reason": "bank_timeout",
        "confidence": 0.8,
        "retry_success_prob": 0.7,
    }
    ml.update(overrides)
    return ml


def test_max_retries_stops():
    txn = base_txn(previous_failures=MAX_RETRIES)
    decision = decide(base_ml(), txn)
    assert decision["action"] == "STOP"
    assert decision["source"] == "hard_guardrail"


def test_below_max_retries_does_not_stop():
    txn = base_txn(previous_failures=MAX_RETRIES - 1)
    decision = decide(base_ml(confidence=0.9), txn)
    assert decision["action"] != "STOP"


def test_card_expired_never_retried():
    decision = decide(base_ml(predicted_reason="card_expired"), base_txn())
    assert decision["action"] == "NUDGE_CUSTOMER"
    assert "expired" in decision["customer_message"].lower()
    assert decision["source"] == "hard_guardrail"


def test_amount_over_cap_escalates():
    txn = base_txn(amount=AUTO_ACTION_AMOUNT_CAP + 1)
    decision = decide(base_ml(confidence=0.95), txn)
    assert decision["action"] == "ESCALATE_TO_HUMAN"
    assert decision["source"] == "hard_guardrail"


def test_low_confidence_escalates():
    decision = decide(base_ml(confidence=LOW_CONFIDENCE_THRESHOLD - 0.01), base_txn())
    assert decision["action"] == "ESCALATE_TO_HUMAN"
    assert decision["source"] == "hard_guardrail"


def test_hard_guardrail_priority_over_everything_else():
    # Even a scenario the mock layer would happily RETRY_NOW on gets
    # stopped once prior failures hit the cap — guardrails run first
    # and unconditionally short-circuit.
    txn = base_txn(previous_failures=MAX_RETRIES)
    decision = decide(
        base_ml(predicted_reason="network_error", retry_success_prob=0.95, confidence=0.95),
        txn,
    )
    assert decision["action"] == "STOP"


def test_mock_insufficient_funds_waits_for_salary_credit():
    decision = decide(
        base_ml(predicted_reason="insufficient_funds", confidence=0.9), base_txn()
    )
    assert decision["action"] == "RETRY_DELAYED"
    assert decision["wait_hours"] == 48
    assert decision["source"] == "mock_llm"


def test_mock_high_retry_prob_retries_now():
    decision = decide(
        base_ml(predicted_reason="network_error", retry_success_prob=0.8, confidence=0.9),
        base_txn(),
    )
    assert decision["action"] == "RETRY_NOW"


def test_mock_moderate_retry_prob_nudges():
    decision = decide(
        base_ml(predicted_reason="otp_failed", retry_success_prob=0.4, confidence=0.9),
        base_txn(),
    )
    assert decision["action"] == "NUDGE_CUSTOMER"


def test_mock_low_retry_prob_escalates():
    decision = decide(
        base_ml(predicted_reason="risk_block", retry_success_prob=0.1, confidence=0.9),
        base_txn(),
    )
    assert decision["action"] == "ESCALATE_TO_HUMAN"
