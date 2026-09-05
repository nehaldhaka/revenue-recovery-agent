"""
decision/ev_optimizer.py
-------------------------
Expected-Value action selection, priced against real payment
economics instead of arbitrary flat constants.

Every candidate action is costed in rupees and we pick whichever
maximizes expected net recovered revenue:
  - retries cost real MDR (merchant discount rate) + a flat processor
    fee, the same way a genuine retry attempt would be billed
  - repeated retries on an already-failing customer carry a RISK
    REVIEW penalty: card networks and issuers flag accounts hit with
    unusual retry volume for fraud review, which is a real cost to
    Razorpair-style aggregators (it hurts their own chargeback/fraud
    ratio with the networks), not just an abstract "annoyance" cost
  - NUDGE/ESCALATE use reason-specific recovery-rate priors, since
    the ML model's retry_success_prob specifically measures "will an
    immediate/delayed retry succeed", not "will a human/nudge succeed"

Cost assumptions are intentionally explicit and tunable — in a real
deployment MDR_RATE and the review-flag rate would come from Finance
and Risk respectively.
"""

# ---- Real payment economics (INR / fractions) ----
MDR_RATE = 0.019          # ~1.9% blended MDR assumption per retry attempt processed
MDR_FLAT_FEE = 0.50       # flat gateway fee per attempt, independent of amount
COST_SMS_NUDGE = 0.35     # per SMS/WhatsApp nudge sent
COST_HUMAN_REVIEW = 45.0  # blended agent cost per escalated ticket

# Probability an account gets flagged for risk/fraud review by the
# issuer or card network per PRIOR failure already on the transaction.
# Being flagged doesn't just cost this transaction — it damages
# Razorpay's own chargeback/fraud ratio with the networks, which is
# why this scales with prior_failures rather than being a flat number.
RISK_FLAG_RATE_PER_PRIOR_FAILURE = 0.04
RISK_FLAG_COST_FRACTION_OF_AMOUNT = 0.02  # modeled cost if flagged, as % of amount

NUDGE_RESPONSE_RATE = {
    "insufficient_funds": 0.30,
    "otp_failed": 0.55,
    "card_expired": 0.45,
    "risk_block": 0.10,
    "bank_timeout": 0.20,
    "network_error": 0.20,
    "issuer_down": 0.15,
}

HUMAN_RECOVERY_RATE = {
    "insufficient_funds": 0.40,
    "bank_timeout": 0.75,
    "otp_failed": 0.60,
    "card_expired": 0.55,
    "risk_block": 0.65,
    "network_error": 0.78,
    "issuer_down": 0.55,
}


def _retry_cost(amount: float) -> float:
    """Real MDR + flat fee — what a retry attempt actually costs to
    process, whether or not it succeeds."""
    return amount * MDR_RATE + MDR_FLAT_FEE


def _risk_flag_penalty(amount: float, prior_failures: int) -> float:
    """Expected cost of an additional retry attempt getting the
    account flagged for risk/fraud review, scaled by how many prior
    failures already exist (repeated attempts on the same card are
    exactly the pattern issuers watch for)."""
    flag_prob = min(0.9, RISK_FLAG_RATE_PER_PRIOR_FAILURE * prior_failures)
    return flag_prob * amount * RISK_FLAG_COST_FRACTION_OF_AMOUNT


def score_actions(ml: dict, txn: dict) -> dict:
    """
    Returns {"ev": {action: expected_value_inr, ...}, "best_action": str}.

    Guardrails (card_expired, MAX_RETRIES, low confidence, amount cap)
    and pattern detectors (circuit breaker, fraud guard) should still
    run BEFORE this in decision_maker.decide() — this function only
    scores among actions that are actually still on the table.
    """
    amount = txn["amount"]
    reason = ml["predicted_reason"]
    p_retry = ml["retry_success_prob"]
    prior_failures = txn.get("previous_failures", 0)

    retry_cost = _retry_cost(amount)
    risk_penalty = _risk_flag_penalty(amount, prior_failures)

    ev = {
        "RETRY_NOW": p_retry * amount - retry_cost - risk_penalty,
        "RETRY_DELAYED": p_retry * amount - retry_cost - (risk_penalty * 0.5),
        "NUDGE_CUSTOMER": NUDGE_RESPONSE_RATE.get(reason, 0.25) * amount - COST_SMS_NUDGE,
        "ESCALATE_TO_HUMAN": HUMAN_RECOVERY_RATE.get(reason, 0.45) * amount - COST_HUMAN_REVIEW,
        "STOP": 0.0,
    }
    best_action = max(ev, key=ev.get)
    return {"ev": {k: round(v, 2) for k, v in ev.items()}, "best_action": best_action}


def decide_by_ev(ml: dict, txn: dict) -> dict:
    result = score_actions(ml, txn)
    action = result["best_action"]
    ev = result["ev"]

    wait_hours = 48 if action == "RETRY_DELAYED" else 0
    customer_message = ""
    if action == "NUDGE_CUSTOMER":
        from . import nudge_copy
        customer_message = nudge_copy.generate(ml["predicted_reason"], txn)

    alternatives = ", ".join(
        f"{a}: Rs{v:,.0f}" for a, v in sorted(ev.items(), key=lambda x: -x[1]) if a != action
    )

    return {
        "action": action,
        "wait_hours": wait_hours,
        "customer_message": customer_message,
        "reasoning": (
            f"[EV] Chose {action} — expected net recovery Rs {ev[action]:,.2f} "
            f"(retry cost model: {MDR_RATE:.1%} MDR + Rs{MDR_FLAT_FEE} flat fee, "
            f"risk-flag penalty Rs{_risk_flag_penalty(txn['amount'], txn.get('previous_failures', 0)):,.2f}) "
            f"vs alternatives {alternatives}."
        ),
        "expected_value_breakdown": ev,
    }