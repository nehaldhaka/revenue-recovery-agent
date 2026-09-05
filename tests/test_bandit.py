"""
tests/test_bandit.py
-----------------------
Covers decision/bandit.py: arm-key bucketing, Thompson-Sampling action
selection, and the Bayesian posterior update on reported outcomes.
Uses the same temp-DB fixture pattern as the other new-module tests.
"""
import threading

import pytest

from execution import db
from decision import bandit


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_audit.db"))
    monkeypatch.setattr(db, "_local", threading.local())
    yield db


class TestAmountBucket:
    @pytest.mark.parametrize("amount,expected", [
        (100, "xs"),
        (1000, "s"),
        (3000, "m"),
        (10000, "l"),
        (50000, "xl"),
    ])
    def test_bucket_boundaries(self, amount, expected):
        assert bandit._amount_bucket(amount) == expected


class TestArmKey:
    def test_key_format(self):
        key = bandit.arm_key("RETRY_NOW", "bank_timeout", 2000)
        assert key == "RETRY_NOW|bank_timeout|m"

    def test_different_amounts_in_same_bucket_share_a_key(self):
        assert bandit.arm_key("RETRY_NOW", "bank_timeout", 2000) == \
               bandit.arm_key("RETRY_NOW", "bank_timeout", 4999)


class TestChooseAction:
    def test_returns_one_of_the_candidate_actions(self, fresh_db):
        result = bandit.choose_action("bank_timeout", 2000)
        assert result["chosen_action"] in bandit.CANDIDATE_ACTIONS

    def test_samples_present_for_every_candidate(self, fresh_db):
        result = bandit.choose_action("bank_timeout", 2000)
        assert set(result["samples"].keys()) == set(bandit.CANDIDATE_ACTIONS)

    def test_strong_prior_dominates_thompson_sampling(self, fresh_db):
        # Give RETRY_NOW a very strong success history (99 recovered, 1
        # failed) for this exact reason/amount bucket, and leave every
        # other action at the uniform Beta(1,1) prior. RETRY_NOW should
        # win the large majority of draws. Thompson Sampling is inherently
        # probabilistic, so we seed the RNG for a deterministic, non-flaky
        # CI run rather than relying on an unseeded random draw.
        import random
        random.seed(42)

        for _ in range(99):
            bandit.record_outcome("RETRY_NOW", "bank_timeout", 2000, recovered=True)
        bandit.record_outcome("RETRY_NOW", "bank_timeout", 2000, recovered=False)

        wins = sum(
            1 for _ in range(20)
            if bandit.choose_action("bank_timeout", 2000)["chosen_action"] == "RETRY_NOW"
        )
        assert wins >= 16  # strong majority; a few losses to exploration noise are expected


class TestRecordOutcome:
    def test_first_recovered_outcome_sets_alpha_to_two(self, fresh_db):
        # Beta(1,1) prior -> one success -> Beta(2,1)
        result = bandit.record_outcome("NUDGE_CUSTOMER", "otp_failed", 500, recovered=True)
        assert result["alpha"] == 2.0
        assert result["beta"] == 1.0
        assert result["pulls"] == 1

    def test_first_failed_outcome_sets_beta_to_two(self, fresh_db):
        result = bandit.record_outcome("NUDGE_CUSTOMER", "otp_failed", 500, recovered=False)
        assert result["alpha"] == 1.0
        assert result["beta"] == 2.0
        assert result["pulls"] == 1

    def test_outcomes_accumulate_across_calls(self, fresh_db):
        bandit.record_outcome("ESCALATE_TO_HUMAN", "risk_block", 8000, recovered=True)
        bandit.record_outcome("ESCALATE_TO_HUMAN", "risk_block", 8000, recovered=True)
        result = bandit.record_outcome("ESCALATE_TO_HUMAN", "risk_block", 8000, recovered=False)
        assert result["alpha"] == 3.0   # prior 1 + 2 successes
        assert result["beta"] == 2.0    # prior 1 + 1 failure
        assert result["pulls"] == 3

    def test_different_arms_stay_independent(self, fresh_db):
        # Recording outcomes for one (action, reason, bucket) must not
        # touch the posterior for a different arm.
        bandit.record_outcome("RETRY_NOW", "bank_timeout", 2000, recovered=True)
        arm = db.get_bandit_arm(bandit.arm_key("RETRY_NOW", "insufficient_funds", 2000))
        assert arm is None  # untouched — no arm has been created for it yet

    def test_estimated_success_rate_matches_alpha_beta_ratio(self, fresh_db):
        for _ in range(7):
            bandit.record_outcome("RETRY_NOW", "network_error", 1500, recovered=True)
        for _ in range(3):
            bandit.record_outcome("RETRY_NOW", "network_error", 1500, recovered=False)
        key = bandit.arm_key("RETRY_NOW", "network_error", 1500)
        arm = db.get_bandit_arm(key)
        # alpha = 1 + 7 = 8, beta = 1 + 3 = 4 -> mean = 8 / 12
        est_rate = arm["alpha"] / (arm["alpha"] + arm["beta"])
        assert est_rate == pytest.approx(8 / 12)