"""
retrain.py
-----------
Closes the human-in-the-loop feedback loop.

Pulls every operator-reviewed case (`reviewed=1`) from the audit DB,
treats the operator's `corrected_reason` as ground truth, merges it
with the original synthetic training set, and retrains a *candidate*
Detective model. The candidate is only promoted to
outputs/detective_model.joblib if it beats (or ties, within
--tolerance) the currently-deployed model on a held-out test split —
this is the "shadow deployment" pattern: evaluate before you swap.

Two ways to run it:

  1. Manually, or on a cron:
       python retrain.py                  # always retrain + evaluate
       python retrain.py --auto           # only retrain if the rolling
                                           # override rate crossed
                                           # --threshold in the last
                                           # --window-days

  2. From the API (see main.py's POST /retrain/run), which calls
     run_retrain() directly and, if the candidate is promoted,
     hot-swaps it into the running server without a restart.

Note on the retry-success model: there's no ground-truth "did the
retry actually succeed" label for human-reviewed rows — we only know
what the operator decided to do next, not the real-world outcome. So
the retry model is retrained on the synthetic data only, just re-fit
against the new shared bank_encoder so both models in the candidate
bundle agree on feature encoding. Only the reason classifier learns
from human corrections.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from execution import db

MODEL_PATH = os.path.join("outputs", "detective_model.joblib")
ARCHIVE_DIR = os.path.join("outputs", "model_archive")
RETRAIN_LOG_PATH = os.path.join("outputs", "retrain_log.jsonl")
SYNTHETIC_CSV = os.path.join("outputs", "synthetic_failed_payments.csv")

FEATURE_COLS = [
    "amount", "bank_enc", "hour", "previous_failures",
    "is_subscription", "days_since_last_success",
]

DEFAULT_THRESHOLD = 0.15      # rolling override rate that triggers auto-retrain
DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_HUMAN_ROWS = 15   # don't retrain on a handful of corrections
DEFAULT_TOLERANCE = 0.0       # candidate must be >= old accuracy - tolerance


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load_synthetic() -> pd.DataFrame:
    df = pd.read_csv(SYNTHETIC_CSV)
    return df[["amount", "bank", "hour", "previous_failures",
               "is_subscription", "days_since_last_success", "failure_reason"]].copy()


def _load_human_reviewed() -> pd.DataFrame:
    rows = db.human_reviewed_rows()
    cols = ["amount", "bank", "hour", "previous_failures",
            "is_subscription", "days_since_last_success", "failure_reason"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).rename(columns={"corrected_reason": "failure_reason"})
    return df[cols].copy()


# ------------------------------------------------------------------
# Auto-trigger
# ------------------------------------------------------------------

def should_retrain(threshold: float = DEFAULT_THRESHOLD, window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """The auto-trigger: retrain when the rolling override rate — the
    same number the dashboard's drift panel is already computing —
    crosses `threshold`."""
    stats = db.rolling_override_rate(window_days=window_days)
    stats["threshold"] = threshold
    stats["trigger"] = stats["reviewed"] > 0 and stats["override_rate"] >= threshold
    return stats


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fit_shared_encoders(merged: pd.DataFrame):
    bank_encoder = LabelEncoder().fit(merged["bank"])
    reason_encoder = LabelEncoder().fit(merged["failure_reason"])
    return bank_encoder, reason_encoder


def _stratify_safe(y):
    """train_test_split blows up if any class has < 2 members; fall
    back to an unstratified split rather than crashing a retrain job
    over one rare class."""
    counts = pd.Series(y).value_counts()
    return y if counts.min() >= 2 else None


def _eval_old_model(old_bundle: dict, test_df: pd.DataFrame) -> dict:
    """Scores the CURRENTLY DEPLOYED model on the same held-out rows,
    using its own (old) encoders, so the promotion decision is
    apples-to-apples. Rows whose bank or true label the old model has
    never seen can't be scored against it — they're excluded and the
    count is reported, so the comparison stays honest rather than
    silently favoring the candidate."""
    old_reason_model = old_bundle["reason_model"]
    old_bank_encoder = old_bundle["bank_encoder"]
    old_le = old_bundle["label_encoder"]

    known_banks = set(old_bank_encoder.classes_)
    known_reasons = set(old_le.classes_)
    scorable = test_df[test_df["bank"].isin(known_banks) & test_df["failure_reason"].isin(known_reasons)]
    skipped = len(test_df) - len(scorable)

    if scorable.empty:
        return {"accuracy": None, "n": 0, "skipped": skipped}

    d = scorable.copy()
    d["bank_enc"] = old_bank_encoder.transform(d["bank"])
    X_old = d[FEATURE_COLS]
    y_true = old_le.transform(d["failure_reason"])
    y_pred = old_reason_model.predict(X_old)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "n": int(len(d)), "skipped": int(skipped)}


def _log_result(result: dict):
    os.makedirs("outputs", exist_ok=True)
    with open(RETRAIN_LOG_PATH, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


# ------------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------------

def run_retrain(tolerance: float = DEFAULT_TOLERANCE,
                 min_human_rows: int = DEFAULT_MIN_HUMAN_ROWS,
                 test_size: float = 0.2,
                 random_state: int = 42) -> dict:
    """
    Pulls human corrections -> merges with synthetic data -> trains a
    candidate reason model (and a retry model on synthetic data) ->
    evaluates the candidate against the currently-deployed model on
    the same held-out split -> promotes only if it wins.

    Returns a result dict that's both printable from the CLI and
    (once "new_bundle" is popped) JSON-serialisable for the
    /retrain/run API response. "new_bundle" carries the actual
    fitted models so the caller can hot-swap them into a running
    process without a restart.
    """
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        result = {"promoted": False, "reason": f"No deployed model found at {MODEL_PATH}; "
                                                 f"run train_detective.py first."}
        _log_result(result)
        return result

    old_bundle = joblib.load(MODEL_PATH)

    synthetic_df = _load_synthetic()
    human_df = _load_human_reviewed()

    if len(human_df) < min_human_rows:
        result = {
            "promoted": False,
            "reason": f"Only {len(human_df)} human-reviewed rows available; "
                      f"need at least {min_human_rows} before retraining is worthwhile.",
            "human_rows": len(human_df),
        }
        _log_result(result)
        return result

    required_cols = ["amount", "bank", "hour", "previous_failures",
                      "is_subscription", "days_since_last_success", "failure_reason"]
    merged = pd.concat([synthetic_df, human_df], ignore_index=True).dropna(subset=required_cols)

    bank_encoder, reason_encoder = _fit_shared_encoders(merged)
    merged["bank_enc"] = bank_encoder.transform(merged["bank"])
    merged["reason_enc"] = reason_encoder.transform(merged["failure_reason"])

    X = merged[FEATURE_COLS]
    y = merged["reason_enc"]
    strat = _stratify_safe(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    candidate_reason_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=random_state, class_weight="balanced"
    )
    candidate_reason_model.fit(X_train, y_train)
    candidate_pred = candidate_reason_model.predict(X_test)
    candidate_acc = accuracy_score(y_test, candidate_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, candidate_pred, average="weighted", zero_division=0
    )

    # Retry model: synthetic data only (it's the only data with a real
    # would_retry_succeed label), re-encoded with the NEW shared bank_encoder.
    retry_source = pd.read_csv(SYNTHETIC_CSV).copy()
    retry_source["bank_enc"] = bank_encoder.transform(retry_source["bank"])
    Xr = retry_source[FEATURE_COLS]
    yr = retry_source["would_retry_succeed"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=test_size, random_state=random_state, stratify=yr
    )
    candidate_retry_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=random_state, class_weight="balanced"
    )
    candidate_retry_model.fit(Xr_train, yr_train)
    retry_acc = accuracy_score(yr_test, candidate_retry_model.predict(Xr_test))

    # Evaluate the OLD model on the exact same held-out reason-model rows.
    test_df = merged.loc[X_test.index, required_cols]
    old_eval = _eval_old_model(old_bundle, test_df)
    old_acc = old_eval["accuracy"]

    promote = (old_acc is None) or (candidate_acc >= old_acc - tolerance)

    result = {
        "promoted": promote,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "human_rows_used": int(len(human_df)),
        "synthetic_rows_used": int(len(synthetic_df)),
        "candidate": {
            "reason_accuracy": float(candidate_acc),
            "reason_precision_weighted": float(precision),
            "reason_recall_weighted": float(recall),
            "reason_f1_weighted": float(f1),
            "retry_accuracy": float(retry_acc),
            "n_test": int(len(X_test)),
        },
        "deployed_model_eval_on_same_split": old_eval,
        "tolerance": tolerance,
    }

    if promote:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        joblib.dump(old_bundle, os.path.join(ARCHIVE_DIR, f"detective_model_{stamp}.joblib"))

        new_bundle = {
            "reason_model": candidate_reason_model,
            "retry_model": candidate_retry_model,
            "label_encoder": reason_encoder,
            "bank_encoder": bank_encoder,
            "feature_cols": FEATURE_COLS,
        }
        joblib.dump(new_bundle, MODEL_PATH)

        result["reason"] = (
            f"Candidate accuracy {candidate_acc:.3f} >= deployed accuracy {old_acc:.3f} "
            f"(tolerance {tolerance}). Promoted; previous model archived."
            if old_acc is not None else
            f"No prior model could be scored on this split; promoting candidate "
            f"(accuracy {candidate_acc:.3f}) by default."
        )
        result["new_bundle"] = new_bundle  # for in-process hot-swap only; not JSON-serialisable
    else:
        result["reason"] = (
            f"Candidate accuracy {candidate_acc:.3f} did not beat deployed accuracy "
            f"{old_acc:.3f} (tolerance {tolerance}). Kept the deployed model."
        )

    _log_result({k: v for k, v in result.items() if k != "new_bundle"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                         help="Only retrain if the rolling override rate crosses --threshold.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-human-rows", type=int, default=DEFAULT_MIN_HUMAN_ROWS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    if args.auto:
        trigger_stats = should_retrain(threshold=args.threshold, window_days=args.window_days)
        print(json.dumps(trigger_stats, indent=2))
        if not trigger_stats["trigger"]:
            print(f"Override rate {trigger_stats['override_rate']:.1%} is below "
                  f"the {args.threshold:.0%} threshold over the last {args.window_days}d — skipping retrain.")
            return

    result = run_retrain(tolerance=args.tolerance, min_human_rows=args.min_human_rows)
    result.pop("new_bundle", None)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()