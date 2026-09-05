"""
execution/doer.py
------------------
Worker 3: The Doer.

Takes what the Detective (ML) observed and what the Decision-Maker
(LLM/rules) decided, "executes" that decision, and writes an
audit-trail row so every decision the system makes is explainable and
traceable after the fact.

Execution is still SIMULATED on purpose (see `_simulate_execution`):
no real payment-gateway call, no real SMS/email send. In production,
that's the only function you'd swap out — everything else here (the
audit contract) stays the same.

Two things changed from the original CSV version:

  1. The audit trail now lives in SQLite (execution/db.py) instead of
     a CSV file — see db.py's docstring for why.
  2. Every call is idempotent. Failed-payment webhooks get redelivered
     constantly in real payment systems (at-least-once delivery is the
     norm, not the exception). If the same txn_id fires again inside
     IDEMPOTENCY_WINDOW_SECONDS, we don't write a second audit row or
     "execute" a second time — we return the *first* call's row with
     `duplicate: True` so the caller can tell the difference.
"""
import hashlib
import time
from datetime import datetime, timezone

from execution import db

AUDIT_PATH = db.DB_PATH  # kept for backwards-compat with any external tooling

IDEMPOTENCY_WINDOW_SECONDS = 300  # 5 minutes

# The dashboard's CSS badge classes (see static/dashboard.html) expect
# these short display names. decide() itself returns the more explicit
# names on the left — we only rename for the audit row / API response,
# we never mutate the caller's `decision` dict, so any caller that
# branches on the raw action name keeps working exactly as before.
# main.py already normalises the action to its display name before
# calling us, so this also has to accept display names as a no-op
# pass-through (handled by the .get(..., raw_action) fallback below).
DISPLAY_ACTION = {
    "RETRY_NOW": "RETRY_NOW",
    "RETRY_DELAYED": "RETRY_LATER",
    "NUDGE_CUSTOMER": "NUDGE",
    "ESCALATE_TO_HUMAN": "ESCALATE",
    "STOP": "STOP",
}


def make_idempotency_key(txn: dict, window_seconds: int = IDEMPOTENCY_WINDOW_SECONDS) -> str:
    """
    hash(txn_id + time-window bucket). Bucketing by time (rather than
    hashing txn_id alone) means the *same* txn_id can legitimately be
    reprocessed later — e.g. a real retry attempt an hour after the
    original failure — while still collapsing back-to-back duplicate
    webhook deliveries for the same failure event into one audit row.
    """
    bucket = int(time.time() // window_seconds)
    raw = f"{txn.get('txn_id')}:{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _simulate_execution(action: str, txn: dict, decision: dict) -> str:
    """
    Simulated execution layer. In a real system this is where you'd
    call the retry API, enqueue a delayed job, send an SMS/email/
    WhatsApp nudge, or open a ticket for a human reviewer. For the
    demo we just record what *would* happen.

    Accepts either the Decision-Maker's raw action name
    (RETRY_DELAYED, NUDGE_CUSTOMER, ESCALATE_TO_HUMAN, ...) or the
    dashboard's short display name (RETRY_LATER, NUDGE, ESCALATE, ...)
    since callers may pass either form.
    """
    txn_id = txn.get("txn_id", "unknown")

    if action in ("RETRY_NOW",):
        return f"Fired an immediate retry for {txn_id}."
    if action in ("RETRY_DELAYED", "RETRY_LATER"):
        hours = decision.get("wait_hours", 0)
        return f"Scheduled a delayed retry for {txn_id} in {hours}h."
    if action in ("NUDGE_CUSTOMER", "NUDGE"):
        return f"Sent a customer nudge message for {txn_id}."
    if action in ("ESCALATE_TO_HUMAN", "ESCALATE"):
        return f"Opened a human-review ticket for {txn_id}."
    if action == "STOP":
        return f"Stopped retry attempts for {txn_id}; no further action taken."
    return f"No-op for {txn_id} (unrecognised action '{action}')."


def execute_and_log(txn: dict, ml: dict, decision: dict) -> dict:
    """
    txn: the raw transaction dict (txn_id, amount, bank, hour, ...)
    ml:  the Detective's output (predicted_reason, confidence, retry_success_prob)
    decision: the Decision-Maker's output (action, wait_hours,
               customer_message, reasoning/reason, source)

    Returns the row that was (or, for a duplicate, already had been)
    written to the audit trail, with an added `duplicate` bool.
    """
    idem_key = make_idempotency_key(txn)

    existing = db.find_by_idempotency_key(idem_key)
    if existing is not None:
        existing = dict(existing)
        existing["duplicate"] = True
        return existing

    raw_action = decision.get("action", "STOP")
    display_action = DISPLAY_ACTION.get(raw_action, raw_action)
    reasoning = decision.get("reasoning") or decision.get("reason") or ""
    execution_note = _simulate_execution(raw_action, txn, decision)

    row = {
        "idempotency_key": idem_key,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "txn_id": txn.get("txn_id"),
        "amount": txn.get("amount"),
        "bank": txn.get("bank"),
        "hour": txn.get("hour"),
        "previous_failures": txn.get("previous_failures", 0),
        "is_subscription": txn.get("is_subscription", 0),
        "days_since_last_success": txn.get("days_since_last_success"),
        "predicted_reason": ml.get("predicted_reason"),
        "confidence": ml.get("confidence"),
        "retry_success_prob": ml.get("retry_success_prob"),
        "action": display_action,
        "wait_hours": decision.get("wait_hours", 0),
        "customer_message": decision.get("customer_message") or "",
        "reasoning": reasoning,
        "source": decision.get("source", ""),
        "execution_note": execution_note,
        "duplicate_of": None,
    }

    saved = db.insert_row(row)
    saved["duplicate"] = False
    return saved
