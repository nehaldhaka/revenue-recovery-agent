"""
decision/circuit_breaker.py
----------------------------
Per-bank circuit breaker.

If a bank's recent transactions are dominated by bank-side failure
reasons (bank_timeout, issuer_down, network_error) above a threshold,
we stop recommending retries against that bank and escalate instead —
the same idea as a circuit breaker in front of a struggling
downstream service: stop hammering it, fail fast, and let a human (or
a later health check) decide when to close the circuit again.

This reads straight from the audit trail (execution/db.py) rather
than maintaining a separate set of counters, so "recent" always
matches what's actually in the ledger and there's only one source of
truth for transaction history.

NEW: when the breaker trips, it also asks smart_routing.py which
OTHER bank currently has the best recovery track record for the same
failure reason, and attaches that as `suggested_route` — turning
"stop hammering this bank" into "stop hammering this bank, and here's
where to send traffic instead."
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

from execution import db
from . import smart_routing

WINDOW_MINUTES = 15
MIN_SAMPLES = 5                # don't trip on noise from a handful of requests
FAILURE_RATE_THRESHOLD = 0.6   # >=60% bank-side failures in the window trips it

BANK_SIDE_REASONS = {"bank_timeout", "issuer_down", "network_error"}


def check(bank: Optional[str]) -> Optional[dict]:
    """
    Returns a guardrail-shaped decision dict if the breaker is open for
    `bank` (mirroring _hard_guardrails' return shape so decide() can
    treat it identically), or None if the breaker is closed / there
    isn't enough recent history to evaluate it — meaning "proceed as
    normal".
    """
    if not bank:
        return None

    since = (datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)).isoformat()
    rows = db.recent_for_bank(bank, since)

    if len(rows) < MIN_SAMPLES:
        return None

    bank_side_reasons_seen = [r.get("predicted_reason") for r in rows if r.get("predicted_reason") in BANK_SIDE_REASONS]
    bank_side = len(bank_side_reasons_seen)
    rate = bank_side / len(rows)

    if rate < FAILURE_RATE_THRESHOLD:
        return None

    majority_reason = Counter(bank_side_reasons_seen).most_common(1)[0][0] if bank_side_reasons_seen else None
    route = smart_routing.suggest_route(bank, majority_reason) if majority_reason else {
        "recommended_bank": None, "recovery_rate": None, "candidates": {}
    }

    route_note = ""
    if route["recommended_bank"]:
        route_note = (
            f" Smart routing suggests {route['recommended_bank']} instead "
            f"(recent recovery rate {route['recovery_rate']:.0%} for the same reason)."
        )

    return {
        "action": "ESCALATE_TO_HUMAN",
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": (
            f"Circuit breaker open for {bank}: {bank_side}/{len(rows)} "
            f"transactions in the last {WINDOW_MINUTES}m were bank-side "
            f"failures ({rate:.0%}, threshold {FAILURE_RATE_THRESHOLD:.0%}). "
            f"Escalating instead of continuing to hammer a struggling downstream."
            f"{route_note}"
        ),
        "source": "circuit_breaker",
        "suggested_route": route,
    }