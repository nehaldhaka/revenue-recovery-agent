import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

N = 1500
BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "YesBank", "IDFC"]

BASE = {
    "insufficient_funds": 0.30,
    "bank_timeout": 0.18,
    "card_expired": 0.08,
    "risk_block": 0.13,
    "otp_failed": 0.12,
    "network_error": 0.11,
    "issuer_down": 0.08
}

def generate_one():
    amount = round(np.random.lognormal(mean=6.4, sigma=1.05), 2)
    bank = np.random.choice(BANKS)
    hour = np.random.randint(0, 24)
    prev_fails = np.random.choice([0, 1, 2, 3, 4], p=[0.52, 0.26, 0.13, 0.06, 0.03])
    is_sub = np.random.choice([0, 1], p=[0.68, 0.32])
    days_since = np.random.randint(1, 75)

    probs = BASE.copy()

    if amount > 4500:
        probs["insufficient_funds"] *= 1.7
        probs["risk_block"] *= 1.5
    if hour <= 5 or hour >= 23:
        probs["bank_timeout"] *= 1.9
        probs["issuer_down"] *= 1.6
    if prev_fails >= 2:
        probs["risk_block"] *= 2.1
        probs["otp_failed"] *= 1.6
    if bank in ["YesBank", "IDFC"]:
        probs["bank_timeout"] *= 1.5
        probs["issuer_down"] *= 1.7

    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    reason = np.random.choice(list(probs.keys()), p=list(probs.values()))

    if reason in ["bank_timeout", "network_error", "issuer_down"]:
        p = 0.72
    elif reason == "insufficient_funds":
        p = 0.38 if hour < 10 else 0.58
    elif reason == "otp_failed":
        p = 0.42
    elif reason == "card_expired":
        p = 0.06
    else:
        p = 0.22

    would_succeed = 1 if np.random.rand() < p else 0

    return {
        "txn_id": "txn_" + str(np.random.randint(100000, 999999)),
        "timestamp": (datetime.now() - timedelta(days=np.random.randint(0, 45))).isoformat(),
        "amount": amount,
        "bank": bank,
        "hour": hour,
        "previous_failures": prev_fails,
        "is_subscription": is_sub,
        "days_since_last_success": days_since,
        "failure_reason": reason,
        "would_retry_succeed": would_succeed
    }


if __name__ == "__main__":
    df = pd.DataFrame([generate_one() for _ in range(N)])
    df.to_csv("outputs/synthetic_failed_payments.csv", index=False)
    print("Generated", len(df), "rows")
    print(df["failure_reason"].value_counts(normalize=True).round(3))
