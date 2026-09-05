"""
decision/smart_routing.py
---------------------------
Failover routing suggestion — the "route around the outage" half of
what circuit_breaker.py already detects.

When circuit_breaker.check() trips for a bank (too many bank-side
failures in the recent window), the natural next question is "given
we're pulling traffic off this bank, where SHOULD future attempts for
this failure reason go instead?" This looks at every OTHER bank's
recent recovery rate for the SAME reason (using the real outcome data
recorded via POST /outcome/{txn_id}) and recommends whichever
alternate currently has the best track record — the same instinct
behind intelligent multi-PSP routing in payment aggregation.

Advisory only: it attaches a suggested_route field to the circuit
breaker's decision dict. It never bypasses a guardrail or changes a
financial field.
"""
from datetime import datetime, timedelta, timezone
from execution import db

LOOKBACK_MINUTES = 60
MIN_ROUTE_SAMPLES = 3  # don't recommend a route backed by too little evidence

ALL_BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "YesBank", "IDFC"]


def suggest_route(failing_bank: str, reason: str) -> dict:
    """
    Returns:
      {
        "recommended_bank": str | None,
        "recovery_rate": float | None,
        "candidates": {bank: recovery_rate, ...}
      }
    comparing recent recovery rates for every OTHER bank, filtered to
    the same failure reason so the comparison stays apples-to-apples.
    Returns recommended_bank=None if there isn't enough outcome data
    yet to recommend anything responsibly.
    """
    since = (datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)).isoformat()
    candidates = {}

    for bank in ALL_BANKS:
        if bank == failing_bank:
            continue
        rows = db.recent_for_bank(bank, since)
        relevant = [
            r for r in rows
            if r.get("predicted_reason") == reason and r.get("outcome") in ("recovered", "failed")
        ]
        if len(relevant) < MIN_ROUTE_SAMPLES:
            continue
        recovered = sum(1 for r in relevant if r["outcome"] == "recovered")
        candidates[bank] = round(recovered / len(relevant), 3)

    if not candidates:
        return {"recommended_bank": None, "recovery_rate": None, "candidates": {}}

    best_bank = max(candidates, key=candidates.get)
    return {"recommended_bank": best_bank, "recovery_rate": candidates[best_bank], "candidates": candidates}