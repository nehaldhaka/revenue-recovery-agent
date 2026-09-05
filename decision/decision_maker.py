"""
decision_maker.py
------------------
Takes the ML layer's output (predicted_reason, confidence, retry_success_prob)
plus the raw transaction fields, and decides what to do next.

Hard guardrails run first and cannot be overridden by any later layer,
because these are the non-negotiable safety rules:
  - never retry a card_expired transaction
  - stop after MAX_RETRIES
  - escalate instead of guessing when confidence is too low
  - escalate instead of auto-acting on very large amounts

After guardrails, two pattern-detectors run against the audit trail
(bank-side outage clustering + smart-routing suggestion, then
card-testing/fraud clustering) before handing off to the actual
decision layer — mock rules, a real LLM, the expected-value optimizer,
or the Thompson-Sampling bandit — selected via DECISION_MODE, or
overridden per-request via decide()'s mode_override argument.
"""

import os
import json
import re

from . import circuit_breaker
from . import fraud_guard
from . import ev_optimizer
from . import bandit
from . import nudge_copy

# mock | llm | ev | bandit — server-wide default when no per-request
# override is given.
DECISION_MODE = os.environ.get("DECISION_MODE", "mock")

MAX_RETRIES = 3
# Calibrated against the actual confidence distribution of the reason model
# (median confidence ~0.25 on this 7-class problem) rather than an arbitrary
# round number — otherwise the threshold silently escalates almost
# everything and the "escalate when uncertain" rule stops being meaningful.
LOW_CONFIDENCE_THRESHOLD = 0.20
AUTO_ACTION_AMOUNT_CAP = 20000  # tune to your amount distribution

SYSTEM_PROMPT = """You are the decision layer of a payment recovery agent.
Given an ML diagnosis (predicted failure reason, confidence, retry success
probability) and transaction context, choose exactly one action:

RETRY_NOW, RETRY_DELAYED, NUDGE_CUSTOMER, ESCALATE_TO_HUMAN, STOP

Rules:
- Never suggest retrying "card_expired" transactions.
- If reason is "insufficient_funds", prefer RETRY_DELAYED (customer may be
  waiting on salary credit) over an immediate retry.
- If you are not confident, prefer ESCALATE_TO_HUMAN over guessing.
- Keep any customer_message short and friendly, never pushy.

Respond ONLY with JSON, no markdown fences:
{
  "action": "...",
  "wait_hours": <int>,
  "customer_message": "...",
  "reasoning": "..."
}
"""


def _hard_guardrails(ml: dict, txn: dict):
    reason = ml["predicted_reason"]
    confidence = ml["confidence"]
    amount = txn["amount"]
    prior_failures = txn.get("previous_failures", 0)

    if prior_failures >= MAX_RETRIES:
        return {
            "action": "STOP",
            "wait_hours": 0,
            "customer_message": "",
            "reasoning": f"Hit the hard cap of {MAX_RETRIES} retries. Stopping to avoid spamming the customer.",
            "source": "hard_guardrail",
        }

    if reason == "card_expired":
        return {
            "action": "NUDGE_CUSTOMER",
            "wait_hours": 0,
            "customer_message": nudge_copy.generate("card_expired", txn),
            "reasoning": "card_expired can never succeed on retry; only a customer action fixes this.",
            "source": "hard_guardrail",
        }

    if amount > AUTO_ACTION_AMOUNT_CAP:
        return {
            "action": "ESCALATE_TO_HUMAN",
            "wait_hours": 0,
            "customer_message": "",
            "reasoning": f"Amount {amount:,.0f} exceeds the auto-action cap of {AUTO_ACTION_AMOUNT_CAP:,}. Routing to a human reviewer.",
            "source": "hard_guardrail",
        }

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "action": "ESCALATE_TO_HUMAN",
            "wait_hours": 0,
            "customer_message": "",
            "reasoning": f"Confidence ({confidence:.3f}) is below the {LOW_CONFIDENCE_THRESHOLD:.3f} threshold. Escalating rather than guessing.",
            "source": "hard_guardrail",
        }

    return None


def _mock_decision(ml: dict, txn: dict) -> dict:
    reason = ml["predicted_reason"]
    p = ml["retry_success_prob"]

    if reason == "insufficient_funds":
        return {
            "action": "RETRY_DELAYED",
            "wait_hours": 48,
            "customer_message": "",
            "reasoning": f"[MOCK] insufficient_funds — waiting 48h improves odds of the account being funded before retrying.",
        }
    if p >= 0.5:
        return {
            "action": "RETRY_NOW",
            "wait_hours": 0,
            "customer_message": "",
            "reasoning": f"[MOCK] retry_success_prob is high ({p:.2f}) and the failure looks transient, so retry immediately.",
        }
    if p >= 0.35:
        return {
            "action": "NUDGE_CUSTOMER",
            "wait_hours": 0,
            "customer_message": nudge_copy.generate(reason, txn),
            "reasoning": f"[MOCK] moderate retry odds ({p:.2f}); a gentle nudge is less intrusive than auto-retrying.",
        }
    return {
        "action": "ESCALATE_TO_HUMAN",
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": f"[MOCK] retry odds are low ({p:.2f}); escalating for manual review instead of guessing.",
    }


def _call_real_llm(ml: dict, txn: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = f"""ML diagnosis:
- predicted_reason: {ml['predicted_reason']}
- confidence: {ml['confidence']:.2f}
- retry_success_prob: {ml['retry_success_prob']:.2f}

Transaction:
- amount: {txn['amount']}
- bank: {txn['bank']}
- previous_failures: {txn.get('previous_failures', 0)}
- is_subscription: {bool(txn.get('is_subscription', 0))}
- days_since_last_success: {txn.get('days_since_last_success')}

Decide the single best next action."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def decide(ml: dict, txn: dict, mode_override: str = None) -> dict:
    """
    mode_override: when given (e.g. from the Control Room UI's engine
    picker), takes priority over the server-wide DECISION_MODE env var
    for this single decision. None means "use the server default,"
    same behaviour as before this parameter existed — every existing
    caller (batch_eval.py, tests) that doesn't pass it is unaffected.
    """
    guardrail = _hard_guardrails(ml, txn)
    if guardrail is not None:
        return guardrail

    # Breaker runs after the hard guardrails (those are absolute safety
    # rules) but before pattern-detection / the decision layer — no point
    # reasoning about a bank we've already decided to stop hammering.
    breaker = circuit_breaker.check(txn.get("bank"))
    if breaker is not None:
        return breaker

    # Fraud guard runs next, still ahead of the actual decision layer:
    # a card-testing cluster isn't a "how do we recover this revenue"
    # question, it's a "don't act on this at all" question.
    fraud = fraud_guard.check(txn.get("bank"), txn.get("amount"))
    if fraud is not None:
        return fraud

    mode = mode_override or DECISION_MODE

    try:
        if mode == "bandit":
            decision = bandit.decide_by_bandit(ml, txn)
            decision["source"] = "bandit"
        elif mode == "ev":
            decision = ev_optimizer.decide_by_ev(ml, txn)
            decision["source"] = "ev_optimizer"
        elif mode == "llm" and os.environ.get("ANTHROPIC_API_KEY"):
            decision = _call_real_llm(ml, txn)
            decision["source"] = "llm"
        else:
            decision = _mock_decision(ml, txn)
            decision["source"] = "mock_llm"
    except Exception as e:
        decision = {
            "action": "ESCALATE_TO_HUMAN",
            "wait_hours": 0,
            "customer_message": "",
            "reasoning": f"Decision layer failed ({e}); escalating to human as a safe fallback.",
            "source": "error_fallback",
        }
    return decision