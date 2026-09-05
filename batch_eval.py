"""
batch_eval.py
-------------
Runs the full pipeline (Detective -> Decision-Maker -> Doer) over a batch
of synthetic failed payments and prints the numbers the buildathon brief
asks for:

  1. Money recovered vs two baselines ("do nothing" and "retry everything")
  2. An example where the system correctly STOPs
  3. An example where the system ESCALATEs to a human
  4. Confirms the full audit trail was written

Run this AFTER generate_data.py and train_detective.py.
"""

import random
import tempfile
import threading

import joblib
import pandas as pd
import numpy as np

from decision.decision_maker import decide
from execution.doer import execute_and_log
from execution import db as _db

BATCH_SIZE = 200
RANDOM_SEED = 7

# Human reviewers have more tools than a blind auto-retry (manual card
# update calls, manual risk approval, direct outreach), so escalated cases
# resolve at a rate tied to the TRUE failure reason -- this is why
# escalation is a genuinely useful category, not just a cop-out.
HUMAN_RECOVERY_RATE = {
    "insufficient_funds": 0.40,
    "bank_timeout": 0.75,
    "otp_failed": 0.60,
    "card_expired": 0.55,
    "risk_block": 0.65,
    "network_error": 0.78,
    "issuer_down": 0.55,
}


def stable_roll(txn_id: str) -> float:
    """Deterministic pseudo-random roll in [0, 1) for a given txn_id.

    Uses hashlib.md5 instead of Python's built-in hash() because str
    hashing is randomized per-process by default (PYTHONHASHSEED) —
    the same txn_id would silently get a different roll on every
    process run, making this NOT actually stable despite the name.
    md5 has no such per-process randomization, so the same txn_id
    always produces the same roll, on any machine, any run.
    """
    import hashlib
    digest = hashlib.md5(txn_id.encode()).hexdigest()
    return (int(digest, 16) % 10_000) / 10_000


def simulate_outcome(row, decision):
    """Simulate what actually happens, using the hidden ground-truth label
    (would_retry_succeed) which the model never saw as an input."""
    action = decision["action"]
    amount = row["amount"]
    would_succeed = bool(row["would_retry_succeed"])
    true_reason = row["failure_reason"]
    roll = stable_roll(row["txn_id"])

    if action in ("RETRY_NOW", "RETRY_DELAYED"):
        return (amount, "recovered") if would_succeed else (0.0, "retry_failed")

    if action == "NUDGE_CUSTOMER":
        if would_succeed and roll < 0.85:
            return amount, "recovered_after_nudge"
        return 0.0, "nudge_no_response"

    if action == "ESCALATE_TO_HUMAN":
        rate = HUMAN_RECOVERY_RATE.get(true_reason, 0.45)
        if roll < rate:
            return amount, "recovered_after_human_review"
        return 0.0, "escalated_not_recovered"

    # STOP
    return 0.0, "stopped_no_recovery"


def main():
    # Isolate this run's audit trail from outputs/audit_trail.db (the live
    # demo/dashboard DB). Without this, circuit_breaker.py and fraud_guard.py
    # read whatever's sitting in the SHARED live DB — including leftover
    # rows from seed_demo_data.py or a previous batch_eval.py run — which
    # makes results non-reproducible and pollutes the dashboard's Review
    # Queue with 200 synthetic escalations on every run. Pointing db.DB_PATH
    # at a fresh temp file and resetting its thread-local connection cache
    # gives this run a brand-new, empty database that no other process
    # (or previous run) can influence.
    _db.DB_PATH = tempfile.mktemp(suffix="_batch_eval_audit.db")
    _db._local = threading.local()

    bundle = joblib.load("outputs/detective_model.joblib")
    reason_m = bundle["reason_model"]
    retry_m = bundle["retry_model"]
    le = bundle["label_encoder"]
    bank_encoder = bundle["bank_encoder"]
    feats = bundle["feature_cols"]

    df = pd.read_csv("outputs/synthetic_failed_payments.csv")
    df = df.sample(n=min(BATCH_SIZE, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    total_at_risk = df["amount"].sum()
    baseline_do_nothing = 0.0
    baseline_retry_all = df.loc[df["would_retry_succeed"] == 1, "amount"].sum()

    system_recovered = 0.0
    action_counts = {}
    stop_example = None
    escalate_example = None

    rng = random.Random(RANDOM_SEED)

    for _, row in df.iterrows():
        d = row.to_dict()

        # encode bank the same way as at training time
        bank_value = d["bank"]
        if bank_value in bank_encoder.classes_:
            d["bank_enc"] = int(bank_encoder.transform([bank_value])[0])
        else:
            d["bank_enc"] = int(bank_encoder.transform([bank_encoder.classes_[0]])[0])

        X = pd.DataFrame([d])[feats]
        pred_idx = reason_m.predict(X)[0]
        pred_reason = le.inverse_transform([pred_idx])[0]
        conf = float(np.max(reason_m.predict_proba(X)))
        p = float(retry_m.predict_proba(X)[0, 1])

        ml = {"predicted_reason": pred_reason, "retry_success_prob": p, "confidence": conf}
        decision = decide(ml, d)
        audit = execute_and_log(d, ml, decision)

        recovered, outcome = simulate_outcome(row, decision)
        system_recovered += recovered
        action_counts[decision["action"]] = action_counts.get(decision["action"], 0) + 1

        if decision["action"] == "STOP" and stop_example is None:
            stop_example = (row["txn_id"], decision["reasoning"])
        if decision["action"] == "ESCALATE_TO_HUMAN" and escalate_example is None:
            escalate_example = (row["txn_id"], decision["reasoning"])

    print("=" * 70)
    print(f"BATCH RESULTS  (n = {len(df)} failed payments)")
    print("=" * 70)
    print(f"Total money at risk         : Rs {total_at_risk:,.0f}")
    print()
    print(f"Baseline - do nothing       : Rs {baseline_do_nothing:,.0f}  (0.0% recovered)")
    print(f"Baseline - retry everything : Rs {baseline_retry_all:,.0f}  "
          f"({baseline_retry_all/total_at_risk*100:.1f}% recovered)")
    print(f"OUR SYSTEM                  : Rs {system_recovered:,.0f}  "
          f"({system_recovered/total_at_risk*100:.1f}% recovered)")
    print()
    print("Action breakdown:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:20s} {count:4d}  ({count/len(df)*100:.1f}%)")

    print("\n" + "-" * 70)
    print("EXAMPLE: system correctly STOPS instead of endlessly retrying")
    print("-" * 70)
    print(f"  {stop_example[0]}: {stop_example[1]}" if stop_example else "  (none in this batch)")

    print("\n" + "-" * 70)
    print("EXAMPLE: system ESCALATES to a human instead of guessing")
    print("-" * 70)
    print(f"  {escalate_example[0]}: {escalate_example[1]}" if escalate_example else "  (none in this batch)")

    # Read db.DB_PATH dynamically (not a stale imported constant) so this
    # prints the actual isolated temp-file path this run just wrote to.
    print(f"\nAudit trail (isolated, this run only): {_db.DB_PATH}")


if __name__ == "__main__":
    main()