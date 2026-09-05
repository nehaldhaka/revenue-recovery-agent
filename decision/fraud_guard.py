"""
decision/fraud_guard.py
------------------------
Card-testing detector.

Fraudsters validate stolen card numbers by firing many small-amount
transactions against a payment gateway in rapid succession, hoping a
few slip through. That pattern looks nothing like a genuine customer's
failed payment: real failures are amount-varied and spread over time;
card-testing is tight-amount-band, high-frequency, short-window,
low-value.

Right now every failed payment is treated as "how do we recover this
revenue" — but some failures aren't customers at all, they're an
attack. This module distinguishes the two and routes attack-shaped
clusters to a human/fraud-ops review instead of letting the recovery
pipeline try to "help" an attacker succeed.

Reuses the audit trail (execution/db.py) as its source of truth, same
as circuit_breaker.py, so there's no separate counter state to keep
in sync with what's actually been logged.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from execution import db

WINDOW_MINUTES = 5
MIN_ATTEMPTS = 6          # >=6 failures against one bank in 5 min is not organic
LOW_AMOUNT_CEILING = 100  # card-testing txns are typically tiny (Rs 1-100)
AMOUNT_TIGHT_BAND = 0.9   # if stdev/mean of recent amounts is this low, flag it


def check(bank: Optional[str], amount: Optional[float]) -> Optional[dict]:
    """
    Returns a guardrail-shaped decision dict (same shape as
    circuit_breaker.check()) if the current bank shows a card-testing
    pattern in the recent window, or None if it doesn't / there isn't
    enough history to evaluate — meaning "proceed as normal".
    """
    if not bank or amount is None:
        return None

    if amount > LOW_AMOUNT_CEILING * 3:
        return None  # this transaction itself isn't in the suspicious range

    since = (datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)).isoformat()
    rows = db.recent_for_bank(bank, since)

    if len(rows) < MIN_ATTEMPTS:
        return None

    amounts = [r["amount"] for r in rows if r.get("amount") is not None]
    if not amounts:
        return None

    mean_amt = sum(amounts) / len(amounts)
    if mean_amt > LOW_AMOUNT_CEILING:
        return None  # cluster exists but isn't low-value -> not the card-testing shape

    variance = sum((a - mean_amt) ** 2 for a in amounts) / len(amounts)
    stdev = variance ** 0.5
    tightness = stdev / mean_amt if mean_amt else 999

    if tightness <= AMOUNT_TIGHT_BAND:
        return {
            "action": "ESCALATE_TO_HUMAN",
            "wait_hours": 0,
            "customer_message": "",
            "reasoning": (
                f"Fraud guard: {len(rows)} low-value failures against {bank} in "
                f"{WINDOW_MINUTES}m, tight amount band (stdev/mean={tightness:.2f}, "
                f"threshold {AMOUNT_TIGHT_BAND}). Pattern matches card-testing, not "
                f"genuine customer failures. Escalating for fraud review instead of "
                f"retrying/recovering."
            ),
            "source": "fraud_guard",
        }
    return None