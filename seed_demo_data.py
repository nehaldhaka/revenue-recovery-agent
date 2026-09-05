"""
seed_demo_data.py
--------------------
Run this ONCE, right before a demo, so the bandit learning panel,
smart-routing suggestion, and circuit breaker don't look empty/broken
on a cold database. All three depend on recent history — an empty DB
means an empty demo.

This talks directly to execution/db.py and decision/bandit.py (no
running server required), so you can run it standalone:

    python seed_demo_data.py

What it seeds:
  1. Bandit posteriors — ~230 synthetic outcomes across a spread of
     (action, reason, amount-bucket) arms, with deliberately different
     win rates per arm so /bandit/arms shows a real, varied spread
     instead of a flat 50% prior everywhere.
  2. A circuit-breaker trigger — 6 recent bank_timeout failures against
     YesBank, enough to trip decision/circuit_breaker.py's threshold
     the next time a YesBank case is submitted.
  3. Smart-routing comparison data — recovery history for HDFC and
     ICICI on the same bank_timeout reason, so smart_routing.py has
     something to recommend once the YesBank breaker trips.
  4. A fraud-guard trigger — 7 low-value, tightly-clustered failures
     against SBI, enough to trip fraud_guard.py's card-testing detector
     the next time a low-value SBI case is submitted.

Notes:
  - Circuit breaker / fraud-guard windows are short (15m / 5m), so this
    needs to be re-run shortly before you actually demo those two.
  - Bandit posteriors have no time window — seed them anytime.
  - Safe to run multiple times; it only adds rows/updates posteriors,
    never deletes anything.
"""
import random
from datetime import datetime, timezone

from execution import db
from decision import bandit

random.seed(7)  # reproducible seed data run-to-run


# ------------------------------------------------------------------
# 1. Bandit posteriors
# ------------------------------------------------------------------
# (action, reason, amount) -> true win probability used to generate
# synthetic outcomes. Deliberately varied so the learned posteriors
# show a real spread instead of all landing near 50%.
BANDIT_SCENARIOS = [
    ("RETRY_NOW",        "bank_timeout",       1500, 0.78, 40),
    ("RETRY_DELAYED",    "bank_timeout",       1500, 0.55, 25),
    ("NUDGE_CUSTOMER",   "bank_timeout",       1500, 0.30, 15),
    ("ESCALATE_TO_HUMAN","bank_timeout",       1500, 0.70, 10),

    ("RETRY_DELAYED",    "insufficient_funds", 3000, 0.62, 30),
    ("RETRY_NOW",        "insufficient_funds", 3000, 0.25, 20),
    ("NUDGE_CUSTOMER",   "insufficient_funds", 3000, 0.35, 20),

    ("NUDGE_CUSTOMER",   "otp_failed",         800,  0.58, 25),
    ("RETRY_NOW",        "otp_failed",         800,  0.40, 15),

    ("ESCALATE_TO_HUMAN","risk_block",         9000, 0.66, 18),
    ("RETRY_NOW",        "risk_block",         9000, 0.12, 12),
]


def seed_bandit():
    total_outcomes = 0
    for action, reason, amount, win_prob, n in BANDIT_SCENARIOS:
        for _ in range(n):
            recovered = random.random() < win_prob
            bandit.record_outcome(action, reason, amount, recovered)
            total_outcomes += 1
    print(f"[bandit] recorded {total_outcomes} synthetic outcomes across {len(BANDIT_SCENARIOS)} arms.")


# ------------------------------------------------------------------
# 2 & 3. Circuit breaker (YesBank) + smart-routing comparisons
# ------------------------------------------------------------------
def _insert_audit_row(txn_id, bank, amount, reason, action="ESCALATE", outcome=None):
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "idempotency_key": f"seed_{txn_id}",
        "logged_at": now,
        "txn_id": txn_id,
        "amount": amount,
        "bank": bank,
        "hour": datetime.now(timezone.utc).hour,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 5,
        "predicted_reason": reason,
        "confidence": 0.7,
        "retry_success_prob": 0.6,
        "action": action,
        "wait_hours": 0,
        "customer_message": "",
        "reasoning": "seeded demo data",
        "source": "seed_script",
        "execution_note": "",
        "duplicate_of": None,
    }
    # find_by_idempotency_key returns None if this row doesn't exist yet
    existing = db.find_by_idempotency_key(row["idempotency_key"])
    if existing:
        saved = existing
    else:
        saved = db.insert_row(row)
    if outcome is not None:
        db.record_outcome(txn_id, recovered=(outcome == "recovered"))
    return saved


def seed_circuit_breaker_and_routing():
    # YesBank: 6 recent bank_timeout failures -> trips circuit_breaker.py
    # (threshold: >=5 samples, >=60% bank-side reasons in last 15 minutes).
    for i in range(6):
        _insert_audit_row(f"seed_yesbank_{i}", "YesBank", 2500, "bank_timeout")
    print("[circuit_breaker] seeded 6 recent bank_timeout failures for YesBank.")

    # HDFC: strong recovery track record on the same reason (3/4 recovered).
    for i, outcome in enumerate(["recovered", "recovered", "recovered", "failed"]):
        _insert_audit_row(f"seed_hdfc_route_{i}", "HDFC", 2500, "bank_timeout", outcome=outcome)

    # ICICI: weaker recovery track record on the same reason (2/4 recovered),
    # so HDFC should be the recommended route, not ICICI.
    for i, outcome in enumerate(["recovered", "recovered", "failed", "failed"]):
        _insert_audit_row(f"seed_icici_route_{i}", "ICICI", 2500, "bank_timeout", outcome=outcome)

    print("[smart_routing] seeded HDFC (75% recovery) and ICICI (50% recovery) "
          "comparison data for bank_timeout — HDFC should be the recommended route.")


# ------------------------------------------------------------------
# 4. Fraud guard (SBI)
# ------------------------------------------------------------------
def seed_fraud_guard():
    # 7 low-value, tightly-clustered failures against SBI -> trips
    # fraud_guard.py's card-testing detector (threshold: >=6 samples,
    # mean <= Rs100, stdev/mean <= 0.9, within last 5 minutes).
    amounts = [10, 11, 9, 10, 12, 10, 11]
    for i, amt in enumerate(amounts):
        _insert_audit_row(f"seed_sbi_fraud_{i}", "SBI", amt, "risk_block")
    print(f"[fraud_guard] seeded {len(amounts)} tightly-clustered low-value failures for SBI.")


def main():
    print("Seeding demo data into", db.DB_PATH)
    print("-" * 60)
    seed_bandit()
    seed_circuit_breaker_and_routing()
    seed_fraud_guard()
    print("-" * 60)
    print("Done. Suggested demo flow:")
    print("  1. GET /bandit/arms                 -> spread of learned success rates")
    print("  2. POST /recover with bank=YesBank   -> circuit breaker trips + routing suggestion")
    print("  3. POST /recover with bank=SBI, amount<100 -> fraud guard trips")
    print("Note: circuit-breaker/fraud-guard windows are short — re-run this script")
    print("shortly before you actually demo those two.")


if __name__ == "__main__":
    main()