"""
remote_batch_eval.py
-----------------------
Like batch_eval.py, but hits a LIVE deployed server's POST /recover over
HTTP instead of calling decide() in-process against the local model file.
batch_eval.py can only ever tell you what your local machine's model
produces — it never touches the network. This script is what actually
answers "does the DEPLOYED server produce the same recovery numbers my
README claims," which is a genuinely different question.

Usage:
    python remote_batch_eval.py --url https://your-app.onrender.com
    python remote_batch_eval.py --url https://your-app.onrender.com --decision-mode ev

Requires outputs/synthetic_failed_payments.csv to already exist locally
(same file batch_eval.py reads) — this script samples the same 200 rows
with the same RANDOM_SEED, so the transaction set is identical to what
batch_eval.py evaluated locally; only the execution path (HTTP vs
in-process) differs.

Honest caveat, read before comparing numbers:
  Every request this script sends is a REAL write to the deployed
  server's live audit_trail.db (there's no way to run this against
  /recover without that — the endpoint doesn't have a dry-run mode).
  circuit_breaker.py and fraud_guard.py read recent history from that
  same database, so running this script MULTIPLE TIMES against the
  same deployment can cause later runs to trip guardrails that didn't
  fire on the first run, simply because more history has accumulated.
  For the cleanest comparison, run this ONCE against a freshly
  deployed/restarted server, right after a fresh model build.
"""
import argparse
import time

import pandas as pd
import requests

# Reuse the exact same outcome-simulation logic as batch_eval.py so the
# recovered-Rs calculation is identical — just import it rather than
# duplicating the HUMAN_RECOVERY_RATE table and stable_roll() function.
from batch_eval import HUMAN_RECOVERY_RATE, stable_roll, simulate_outcome, BATCH_SIZE, RANDOM_SEED

# main.py normalises decision["action"] to these SHORT display names
# before returning it over HTTP (RETRY_DELAYED -> RETRY_LATER,
# NUDGE_CUSTOMER -> NUDGE, ESCALATE_TO_HUMAN -> ESCALATE) — see
# _run_pipeline() in main.py. But simulate_outcome() (imported from
# batch_eval.py, which calls decide() directly and never sees this
# normalisation) checks against the RAW action names. Without this
# mapping, "RETRY_LATER"/"NUDGE"/"ESCALATE" silently fail to match any
# of simulate_outcome()'s branches and fall through to the STOP
# branch — scoring every retry/nudge/escalation as Rs 0 recovered
# regardless of whether it actually would have succeeded. That would
# make the deployed system look dramatically worse than it is. This
# is the exact same mapping main.py's own POST /outcome/{txn_id}
# handler already uses for the same reason.
ACTION_MAP = {
    "RETRY_LATER": "RETRY_DELAYED",
    "NUDGE": "NUDGE_CUSTOMER",
    "ESCALATE": "ESCALATE_TO_HUMAN",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL of the deployed server, e.g. https://your-app.onrender.com")
    parser.add_argument("--decision-mode", default=None, help="Optional decision_mode override sent per-request (mock/ev/bandit)")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds to sleep between requests (be polite to a free-tier host)")
    args = parser.parse_args()

    recover_url = args.url.rstrip("/") + "/recover"

    df = pd.read_csv("outputs/synthetic_failed_payments.csv")
    df = df.sample(n=min(BATCH_SIZE, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    total_at_risk = df["amount"].sum()
    baseline_do_nothing = 0.0
    baseline_retry_all = df.loc[df["would_retry_succeed"] == 1, "amount"].sum()

    system_recovered = 0.0
    action_counts = {}
    errors = 0
    stop_example = None
    escalate_example = None

    print(f"POSTing {len(df)} transactions to {recover_url} ...")

    for i, row in df.iterrows():
        payload = {
            "txn_id": row["txn_id"],
            "amount": float(row["amount"]),
            "bank": row["bank"],
            "hour": int(row["hour"]),
            "previous_failures": int(row["previous_failures"]),
            "is_subscription": int(row["is_subscription"]),
            "days_since_last_success": int(row["days_since_last_success"]),
        }
        if args.decision_mode:
            payload["decision_mode"] = args.decision_mode

        try:
            resp = requests.post(recover_url, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            action = data["decision"]["action"]
            action = ACTION_MAP.get(action, action)  # display name -> raw name simulate_outcome() expects
            reasoning = data["decision"].get("reasoning") or data["decision"].get("reason") or ""
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(df)}] ERROR on {row['txn_id']}: {e}")
            continue

        recovered, outcome = simulate_outcome(row, {"action": action})
        system_recovered += recovered
        action_counts[action] = action_counts.get(action, 0) + 1

        if action == "STOP" and stop_example is None:
            stop_example = (row["txn_id"], reasoning)
        if action == "ESCALATE_TO_HUMAN" and escalate_example is None:
            escalate_example = (row["txn_id"], reasoning)

        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(df)} done")
        time.sleep(args.delay)

    print("=" * 70)
    print(f"REMOTE BATCH RESULTS  (n = {len(df)} failed payments, {errors} request errors)")
    print(f"Target: {recover_url}" + (f" (decision_mode={args.decision_mode})" if args.decision_mode else ""))
    print("=" * 70)
    print(f"Total money at risk         : Rs {total_at_risk:,.0f}")
    print()
    print(f"Baseline - do nothing       : Rs {baseline_do_nothing:,.0f}  (0.0% recovered)")
    print(f"Baseline - retry everything : Rs {baseline_retry_all:,.0f}  "
          f"({baseline_retry_all/total_at_risk*100:.1f}% recovered)")
    print(f"DEPLOYED SYSTEM             : Rs {system_recovered:,.0f}  "
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

    print("\nCompare the 'DEPLOYED SYSTEM' % above against your README's committed")
    print("recovery number. Same or very close = your local and deployed builds")
    print("agree. Meaningfully different = something drifted (library version,")
    print("model file, or accumulated audit history skewing circuit_breaker/")
    print("fraud_guard) — worth investigating before you present this number.")


if __name__ == "__main__":
    main()