"""
decision/bandit.py
--------------------
Thompson-Sampling multi-armed bandit for action selection.

decision_maker's other modes (mock/llm/ev) pick an action from a
FIXED formula — static probabilities, static costs, chosen once at
launch and never revisited. This mode instead treats each
(action, predicted_reason, amount_bucket) combination as its own
Bernoulli arm with a Beta(alpha, beta) posterior over "does this
action recover the payment." Every time a real outcome is reported
via POST /outcome/{txn_id}, the corresponding arm's posterior updates
— so the system's choices provably improve as more cases run through
it, instead of being frozen at whatever assumptions were hardcoded on
day one.

State lives in execution/db.py's bandit_arms table so it persists
across restarts and is shared across every worker thread.
"""
import random
from execution import db

CANDIDATE_ACTIONS = ["RETRY_NOW", "RETRY_DELAYED", "NUDGE_CUSTOMER", "ESCALATE_TO_HUMAN"]


def _amount_bucket(amount: float) -> str:
    """Coarse amount buckets so the bandit doesn't need millions of
    samples per exact rupee value to start learning something useful."""
    if amount < 500:
        return "xs"
    if amount < 2000:
        return "s"
    if amount < 5000:
        return "m"
    if amount < 15000:
        return "l"
    return "xl"


def arm_key(action: str, reason: str, amount: float) -> str:
    return f"{action}|{reason}|{_amount_bucket(amount)}"


def _sample_arm(action: str, reason: str, amount: float) -> float:
    key = arm_key(action, reason, amount)
    arm = db.get_bandit_arm(key)
    alpha = arm["alpha"] if arm else 1.0  # Beta(1,1) = uniform prior = "no idea yet"
    beta = arm["beta"] if arm else 1.0
    return random.betavariate(alpha, beta)


def choose_action(reason: str, amount: float, candidates=None) -> dict:
    """
    Draws one sample from each candidate arm's current posterior and
    picks the arm with the highest sample (Thompson Sampling). This
    naturally balances exploration (arms with wide/uncertain
    posteriors occasionally win by chance) against exploitation (arms
    with a strong track record usually win) without needing a
    separate epsilon-greedy schedule.
    """
    candidates = candidates or CANDIDATE_ACTIONS
    samples = {a: _sample_arm(a, reason, amount) for a in candidates}
    best = max(samples, key=samples.get)
    return {"chosen_action": best, "samples": {k: round(v, 4) for k, v in samples.items()}}


def record_outcome(action: str, reason: str, amount: float, recovered: bool) -> dict:
    """Bayesian update: a recovered case adds to alpha (success count),
    a failed case adds to beta (failure count). Called from
    POST /outcome/{txn_id} once the real-world result is known."""
    key = arm_key(action, reason, amount)
    arm = db.get_bandit_arm(key)
    alpha = (arm["alpha"] if arm else 1.0) + (1.0 if recovered else 0.0)
    beta = (arm["beta"] if arm else 1.0) + (0.0 if recovered else 1.0)
    pulls = (arm["pulls"] if arm else 0) + 1
    db.upsert_bandit_arm(key, alpha, beta, pulls)
    return {"arm_key": key, "alpha": alpha, "beta": beta, "pulls": pulls}


def decide_by_bandit(ml: dict, txn: dict) -> dict:
    """Full decision dict in the shape decision_maker.decide() and
    doer.execute_and_log() expect."""
    reason = ml["predicted_reason"]
    amount = txn["amount"]
    result = choose_action(reason, amount)
    action = result["chosen_action"]

    wait_hours = 48 if action == "RETRY_DELAYED" else 0
    customer_message = ""
    if action == "NUDGE_CUSTOMER":
        from . import nudge_copy
        customer_message = nudge_copy.generate(reason, txn)

    return {
        "action": action,
        "wait_hours": wait_hours,
        "customer_message": customer_message,
        "reasoning": (
            f"[BANDIT] Thompson-sampled {action} for reason={reason}, "
            f"amount-bucket={_amount_bucket(amount)}. Posterior samples: {result['samples']}."
        ),
        "bandit_samples": result["samples"],
    }