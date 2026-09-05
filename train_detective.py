import platform
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("outputs/synthetic_failed_payments.csv")

# --- encode the bank string into a number the model can use ---
bank_encoder = LabelEncoder()
df["bank_enc"] = bank_encoder.fit_transform(df["bank"])

# --- encode the failure_reason target into a number ---
le = LabelEncoder()
df["reason_enc"] = le.fit_transform(df["failure_reason"])

feature_cols = [
    "amount", "bank_enc", "hour", "previous_failures",
    "is_subscription", "days_since_last_success",
]

X = df[feature_cols]
y_reason = df["reason_enc"]
y_retry = df["would_retry_succeed"]

X_train, X_test, yr_train, yr_test, yt_train, yt_test = train_test_split(
    X, y_reason, y_retry, test_size=0.2, random_state=42, stratify=y_reason
)

reason_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
)
reason_model.fit(X_train, yr_train)
reason_acc = accuracy_score(yr_test, reason_model.predict(X_test))

retry_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
)
retry_model.fit(X_train, yt_train)
retry_acc = accuracy_score(yt_test, retry_model.predict(X_test))

n_classes = y_reason.nunique()
print(f"Reason model accuracy : {reason_acc:.3f}  (random baseline = {1/n_classes:.3f})")
print(f"Retry model accuracy  : {retry_acc:.3f}  (random baseline = 0.500)")

# ------------------------------------------------------------------
# Provenance metadata — so a deployed server can report exactly what
# produced its model, and you can diff that against a local training
# run instead of assuming "should be deterministic" is actually true.
# random_state=42 (twice, above) SHOULD make this reproducible given
# the same generate_data.py output — this manifest is what lets you
# confirm that, rather than just trust it.
# ------------------------------------------------------------------
meta = {
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "reason_accuracy": round(float(reason_acc), 4),
    "retry_accuracy": round(float(retry_acc), 4),
    "n_classes": int(n_classes),
    "n_rows_trained_on": int(len(df)),
    "feature_cols": feature_cols,
    "reason_classes": sorted(le.classes_.tolist()),
    "bank_classes": sorted(bank_encoder.classes_.tolist()),
    "versions": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
}

joblib.dump({
    "reason_model": reason_model,
    "retry_model": retry_model,
    "label_encoder": le,           # decodes reason_enc -> failure_reason string
    "bank_encoder": bank_encoder,  # encodes bank string -> bank_enc
    "feature_cols": feature_cols,
    "meta": meta,                  # NEW: provenance, read by GET /model/info
}, "outputs/detective_model.joblib")

# Also write it standalone so it survives even if someone inspects
# outputs/ without wanting to unpickle the whole joblib bundle.
import json
with open("outputs/model_manifest.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved -> outputs/detective_model.joblib")
print("Saved -> outputs/model_manifest.json")
print(f"\nVersions: scikit-learn {meta['versions']['scikit_learn']}, "
      f"numpy {meta['versions']['numpy']}, pandas {meta['versions']['pandas']}, "
      f"python {meta['versions']['python']}")