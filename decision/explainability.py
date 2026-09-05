"""
decision/explainability.py
-----------------------------
SHAP-based explanation for the Detective's reason classifier.

Automated decisioning in payments increasingly needs to answer "why
did the model say that" for audit/compliance purposes, not just "how
accurate is it on average." This wraps shap.TreeExplainer around the
trained RandomForestClassifier and returns the top-N features that
pushed THIS prediction toward its predicted class, in a form the
dashboard can show next to the raw prediction (e.g. "predicted
bank_timeout mainly because: hour=2am, bank=YesBank").
"""
from typing import List, Dict


def explain_prediction(model, X_row, feature_names: List[str], predicted_class_idx: int, top_n: int = 3) -> List[Dict]:
    """
    X_row: a single-row pandas DataFrame with the exact feature_cols
    used at training/inference time (the same object main.py already
    builds before calling reason_m.predict).

    Returns a list of {"feature": str, "contribution": float} sorted
    by |contribution| descending, for the predicted class only. Returns
    [] (rather than raising) if shap isn't installed, fails to import
    for any other reason, or anything about the explanation fails —
    this is a "nice to have" enrichment, never something that should
    break a live prediction.
    """
    try:
        import shap
    except Exception:
        # Broadened from `except ImportError`. shap's own import chain
        # pulls in numba/llvmlite/scikit-image, and a version mismatch
        # among THOSE transitive dependencies can raise something other
        # than ImportError (verified: a numpy/scikit-image incompatibility
        # in shap's plotting module raised a bare TypeError here, which a
        # narrower `except ImportError` would NOT catch). Since
        # explain_prediction() runs unconditionally on every /recover
        # call, letting any import-time failure of this optional
        # dependency go uncaught would 500 every single request instead
        # of just silently skipping the SHAP chips for this one.
        return []

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)

        # RandomForestClassifier with multiple classes -> shap_values is
        # a list of arrays, one per class. Pick the predicted class.
        if isinstance(shap_values, list):
            row_values = shap_values[predicted_class_idx][0]
        else:
            row_values = shap_values[0]

        pairs = list(zip(feature_names, row_values))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)

        return [
            {"feature": name, "contribution": round(float(val), 4)}
            for name, val in pairs[:top_n]
        ]
    except Exception:
        return []