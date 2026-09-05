import queue as _queue
from typing import Optional

from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

from decision.decision_maker import decide
from decision import explainability
from decision import bandit as bandit_module
from execution.doer import execute_and_log
from execution import db as audit_db
import auth
import queueing
import retrain

app = FastAPI(title="Revenue Recovery Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Load Detective model ----------
bundle = joblib.load("outputs/detective_model.joblib")
reason_m = bundle["reason_model"]
retry_m = bundle["retry_model"]
le = bundle["label_encoder"]
bank_encoder = bundle["bank_encoder"]
feats = bundle["feature_cols"]


class FailedPayment(BaseModel):
    txn_id: str
    amount: float
    bank: str
    hour: int
    previous_failures: int
    is_subscription: int
    days_since_last_success: int
    # Per-request override of DECISION_MODE (mock | llm | ev | bandit).
    # Lets a caller (e.g. remote_batch_eval.py) choose which decision
    # engine handles THIS submission, instead of every request being
    # locked to whatever DECISION_MODE the server happened to be
    # started with.
    decision_mode: Optional[str] = None


class LoginRequest(BaseModel):
    operator: str


class ReviewDecision(BaseModel):
    operator: str
    action: str  # RETRY_NOW | RETRY_LATER | NUDGE | STOP
    corrected_reason: Optional[str] = None


class OutcomeReport(BaseModel):
    recovered: bool


def _require_auth(authorization: Optional[str] = Header(default=None)):
    """
    No-op unless REQUIRE_AUTH=1 is set (see auth.py) — so the app still
    runs zero-config out of the box. When enabled, /recover needs a
    valid Bearer token minted by /auth/login.
    """
    if not auth.REQUIRE_AUTH:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):]
    try:
        return auth.verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _run_pipeline(d: dict) -> dict:
    """
    The Detective -> Decision-Maker -> Doer pipeline. This is the
    function /recover hands off to the worker pool (queueing.py)
    instead of running inline on the request thread.
    """
    # Pull the per-request decision-engine override off the payload
    # before it reaches feature selection (X = ...[feats] would choke
    # on an extra non-numeric column otherwise).
    decision_mode_override = d.pop("decision_mode", None)

    # Encode bank the same way it was encoded during training
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

    # NEW: SHAP explanation for this specific prediction, if shap is
    # installed. Returns [] silently if it isn't — never blocks the
    # actual decision on this being available.
    explanation = explainability.explain_prediction(reason_m, X, feats, int(pred_idx))

    ml = {
        "predicted_reason": pred_reason,
        "retry_success_prob": p,
        "confidence": conf,
        "explanation": explanation,
    }

    decision = decide(ml, d, mode_override=decision_mode_override)

    # --- Normalise action names for the dashboard badges ---
    action = decision.get("action", "STOP")
    if action == "RETRY_DELAYED":
        action = "RETRY_LATER"
    elif action == "NUDGE_CUSTOMER":
        action = "NUDGE"
    elif action == "ESCALATE_TO_HUMAN":
        action = "ESCALATE"
    decision["action"] = action

    # Make sure frontend always receives these keys
    decision["reason"] = decision.get("reason") or decision.get("reasoning") or ""
    decision["source"] = decision.get("source", "mock / rules")
    decision["customer_message"] = decision.get("customer_message")

    audit = execute_and_log(d, ml, decision)

    return {
        "ml_prediction": ml,
        "decision": decision,
        "audit_record": audit,
    }


@app.post("/auth/login")
def login(body: LoginRequest):
    """Issues a signed, expiring token for an operator. There's no
    password because the original demo doesn't have operator
    accounts — see auth.py's docstring for the tradeoff. Set
    REQUIRE_AUTH=1 to make /recover actually check this token."""
    name = (body.operator or "Operator").strip()[:40] or "Operator"
    token = auth.issue_token(name)
    return {"token": token, "operator": name}


@app.post("/recover")
def recover(txn: FailedPayment, _claims=Depends(_require_auth)):
    d = txn.dict()
    try:
        return queueing.submit(_run_pipeline, d, timeout=15.0)
    except _queue.Full:
        raise HTTPException(
            status_code=429,
            detail="Recovery worker pool is at capacity, try again shortly.",
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Recovery pipeline timed out.")


@app.post("/outcome/{txn_id}")
def report_outcome(txn_id: str, body: OutcomeReport):
    """
    Records the REAL, ground-truth outcome of a case: did the chosen
    action (retry/nudge/escalation) actually recover the payment? This
    is what closes the loop for both the bandit (decision/bandit.py)
    and smart routing (decision/smart_routing.py) — without this,
    neither module has anything to learn from.

    In production this would be called by a webhook when the retry
    gateway/PSP confirms success or failure; for the demo, call it
    manually or from batch_eval.py's simulated outcomes.
    """
    row = audit_db.record_outcome(txn_id, body.recovered)
    if row is None:
        raise HTTPException(status_code=404, detail="No audit record found for this txn_id.")

    # Feed the outcome into the bandit's posterior, if enough info is
    # present to compute the arm key (action + predicted_reason + amount).
    action = row.get("action")
    reason = row.get("predicted_reason")
    amount = row.get("amount")
    bandit_update = None
    if action and reason and amount is not None:
        # Map dashboard display names back to the bandit's action space.
        action_map = {"RETRY_LATER": "RETRY_DELAYED", "NUDGE": "NUDGE_CUSTOMER", "ESCALATE": "ESCALATE_TO_HUMAN"}
        bandit_action = action_map.get(action, action)
        bandit_update = bandit_module.record_outcome(bandit_action, reason, amount, body.recovered)

    return {"audit_record": row, "bandit_update": bandit_update}


@app.get("/bandit/arms")
def bandit_arms():
    """Inspection endpoint: every arm's current Beta posterior and
    pull count — useful for a debug panel showing what the bandit has
    learned so far (mean win rate = alpha / (alpha + beta))."""
    arms = audit_db.all_bandit_arms()
    for a in arms:
        a["estimated_success_rate"] = round(a["alpha"] / (a["alpha"] + a["beta"]), 3)
    return arms


@app.get("/")
@app.get("/dashboard")
def dashboard():
    """Serves the control-room UI."""
    return FileResponse("static/dashboard.html")


@app.get("/audit-trail")
def audit_trail_page():
    """Serves the standalone Audit Trail UI."""
    return FileResponse("static/audit.html")


@app.get("/health")
def health():
    return {"status": "Revenue Recovery Agent is live"}


@app.get("/model/info")
def model_info():
    """
    Reports exactly what produced the currently-loaded model: training
    accuracy, class lists, and the scikit-learn/numpy/pandas/Python
    versions it was trained with. Compare this against a local
    train_detective.py run's printed output to confirm the deployed
    model is actually reproducing what you saw locally — "random_state=42
    should make this deterministic" is an assumption; this endpoint is
    how you verify it instead of trusting it.
    """
    return bundle.get("meta", {
        "note": "This model was trained before the meta field existed — "
                 "retrain with the updated train_detective.py to populate this."
    })


@app.get("/audit")
def audit_trail(limit: int = Query(30, ge=1, le=100)):
    """
    Returns the most recent audit rows as JSON (newest first) from
    SQLite so the dashboard can render the live table. Field names
    match what the original CSV-backed endpoint returned, so the
    dashboard/audit-trail UI needs zero changes.
    """
    return audit_db.recent_rows(limit=limit)


@app.get("/review")
def review_page():
    """Serves the human-in-the-loop review queue UI."""
    return FileResponse("static/review.html")


@app.get("/review/queue")
def review_queue(limit: int = Query(50, ge=1, le=200)):
    """Escalated decisions waiting for an operator, oldest first."""
    return audit_db.pending_review(limit=limit)


@app.post("/review/{record_id}/decide")
def review_decide(record_id: int, body: ReviewDecision):
    """
    Records an operator's decision on an escalated case: the final
    action to take, and — critically — the *actual* failure reason if
    the Detective model got it wrong. That correction is the labeled
    training example a real retraining pipeline would consume.
    """
    row = audit_db.mark_reviewed(record_id, body.operator, body.action, body.corrected_reason)
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return row


@app.get("/review/stats")
def review_stats():
    """Daily reviewed/overridden counts — the model-drift panel's data
    source. A rising override_rate means the Detective's predicted
    failure reason is disagreeing with operators more often over
    time."""
    return audit_db.override_stats()


@app.get("/retrain/status")
def retrain_status(
    threshold: float = Query(retrain.DEFAULT_THRESHOLD, ge=0.0, le=1.0),
    window_days: int = Query(retrain.DEFAULT_WINDOW_DAYS, ge=1, le=90),
):
    """
    The rolling override rate the dashboard's drift panel is already
    computing, plus whether it has crossed the auto-retrain threshold.
    A cron job — or a "Retrain now" button — can poll this before
    deciding to call /retrain/run.
    """
    return retrain.should_retrain(threshold=threshold, window_days=window_days)


@app.post("/retrain/run")
def retrain_run(
    auto: bool = Query(False, description="Only retrain if the rolling override rate is over threshold"),
    threshold: float = Query(retrain.DEFAULT_THRESHOLD, ge=0.0, le=1.0),
    window_days: int = Query(retrain.DEFAULT_WINDOW_DAYS, ge=1, le=90),
    _claims=Depends(_require_auth),
):
    """
    Pulls every human correction from the audit trail, retrains a
    candidate Detective model, and shadow-evaluates it against the
    model currently deployed on the same held-out split. If the
    candidate wins, it's promoted to outputs/detective_model.joblib
    AND hot-swapped into this running process — no restart required.
    If auto=true, this is a no-op unless the rolling override rate has
    already crossed `threshold`.
    """
    global reason_m, retry_m, le, bank_encoder, feats

    if auto:
        trigger = retrain.should_retrain(threshold=threshold, window_days=window_days)
        if not trigger["trigger"]:
            return {"promoted": False, "reason": "Rolling override rate below threshold.", "stats": trigger}

    result = retrain.run_retrain()
    new_bundle = result.pop("new_bundle", None)

    if result.get("promoted") and new_bundle is not None:
        reason_m = new_bundle["reason_model"]
        retry_m = new_bundle["retry_model"]
        le = new_bundle["label_encoder"]
        bank_encoder = new_bundle["bank_encoder"]
        feats = new_bundle["feature_cols"]

    return result