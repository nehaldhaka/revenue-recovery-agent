# Revenue Recovery Agent

[![tests](https://github.com/nehaldhaka/revenue-recovery-agent/actions/workflows/test.yml/badge.svg)](https://github.com/nehaldhaka/revenue-recovery-agent/actions/workflows/test.yml)

**Live demo:** [https://your-app-name.onrender.com](https://your-app-name.onrender.com) *(replace with your Render/Railway URL)*

![Revenue Recovery Agent control room demo](docs/demo.gif)
*(replace docs/demo.gif with a ~15s screen recording of the dashboard: submit a failed payment, watch the decision render, click through to the audit trail and review queue)*

An AI agent that catches "leaked" payment revenue — failed card/UPI
payments, abandoned checkouts, missed subscription charges — figures out
*why* each one failed, and decides responsibly what to do about it:
retry, wait and retry, nudge the customer, escalate to a human, or stop.

**In business terms:** on the synthetic distribution this demo ships with
(200 failed payments, ₹179,742 total at risk), running in **expected-value
mode** the agent recovers **₹108,743 (60.5%)** of failed-payment value —
**22% more than a naive "retry everything" baseline** (₹88,916, 49.5%) —
by recognizing when a human reviewer's tools (manual card updates, direct
outreach, risk overrides) are worth more in expectation than another
automated retry, while still auto-**STOP**ping 11% of cases outright to
avoid spamming customers who've already failed 3+ times. See
[Batch Evaluation Results](#batch-evaluation-results) below for the full
run, the mode-by-mode comparison, and an honest look at the tradeoff behind
that number.

Built as three workers, each doing the job it's actually good at:

```
                 ┌──────────────┐      ┌───────────────────┐      ┌───────────────┐
 failed payment  │  DETECTIVE   │      │   DECISION-MAKER   │      │      DOER      │
 ──────────────► │  (ML model)  │ ───► │   (LLM + hard      │ ───► │  (execute +    │
                  │              │      │    guardrails)     │      │   audit log)   │
                  └──────────────┘      └───────────────────┘      └───────────────┘
                  "why did this          "what should we do        "do it, and write
                   fail, and will         about it — and know       down what
                   a retry work?"         when to stop?"            happened & why"
```

| Worker | File | Job |
|---|---|---|
| **Detective** | `generate_data.py`, `train_detective.py` | Two `RandomForestClassifier`s trained on synthetic failed-payment data: one predicts the **failure reason** (7 classes: insufficient funds, bank timeout, card expired, risk block, OTP failed, network error, issuer down), the other predicts the **probability a retry succeeds**. |
| **Decision-Maker** | `decision/decision_maker.py` | Hard, non-negotiable guardrails run first (never retry `card_expired`, stop after `MAX_RETRIES`, escalate below a confidence threshold, escalate above an amount cap, per-bank circuit breaker, card-testing fraud guard). If none fire, the action itself is chosen by whichever mode `DECISION_MODE` is set to: a deterministic mock, a real LLM (Claude, via the Anthropic API), an expected-value optimizer priced against real MDR/risk-flag economics, or a self-improving Thompson-Sampling bandit. |
| **Doer** | `execution/doer.py` | "Executes" the decision (simulated — no real payment gateway call) and appends an idempotent audit-trail row to SQLite (`outputs/audit_trail.db`) with everything: the diagnosis, the decision, the reasoning, and the source (guardrail / circuit breaker / fraud guard / LLM / mock / EV / bandit). |
| **API + Dashboard** | `main.py`, `static/dashboard.html` | FastAPI server exposing `/recover`, `/audit`, `/review`, `/auth/login`, `/outcome/{txn_id}`, `/bandit/arms`, serving the control-room dashboard, audit-trail, and review-queue UIs. |

## Payments-engineering hardening

Beyond the three-worker pipeline, the following are built in to reflect how
this would actually need to behave in front of real payment traffic:

| Concern | What's implemented | Where |
|---|---|---|
| **Duplicate webhooks** | `/recover` is idempotent — a hash of `txn_id` + a 5-minute time bucket dedupes redelivered webhooks so the same failure never gets double-logged (and revenue never double-counted). Firing the same txn twice returns the *first* audit row with `duplicate: true`. Verified under 50 concurrent duplicate deliveries — see [Reliability & Load Testing](#reliability--load-testing) below. | `execution/doer.py` (`make_idempotency_key`) |
| **A bank starts failing hard** | A per-bank circuit breaker watches the last 15 minutes of audit history; if ≥60% of a bank's recent transactions are bank-side failures (`bank_timeout`, `issuer_down`, `network_error`), the breaker trips and escalates instead of continuing to retry against a struggling downstream — and attaches a **smart-routing suggestion** naming whichever other bank currently has the best recent recovery rate for that same failure reason. | `decision/circuit_breaker.py`, `decision/smart_routing.py`, wired into `decision/decision_maker.py::decide()` |
| **Card-testing / fraud clusters** | A separate guard watches for the card-testing fingerprint — many low-value failures against one bank in a tight amount band in a short window — and escalates for fraud review instead of letting the recovery pipeline try to "help" an attacker succeed. | `decision/fraud_guard.py` |
| **Self-improving decisioning** | `DECISION_MODE=bandit` treats each (action, failure reason, amount bucket) as a Thompson-Sampling arm with a Beta posterior, updated from real reported outcomes (`POST /outcome/{txn_id}`) rather than the fixed probabilities the mock/EV modes use. `GET /bandit/arms` exposes the learned posteriors. | `decision/bandit.py` |
| **Cost-aware decisioning** | `DECISION_MODE=ev` prices every candidate action against real payment economics (MDR + flat gateway fee per retry, a risk-flag penalty that scales with prior failures) instead of thresholding `retry_success_prob` against arbitrary cutoffs, and picks whichever action maximizes expected net recovery. **This is the mode behind the headline recovery number above.** | `decision/ev_optimizer.py` |
| **Audit trail storage** | SQLite instead of a growing CSV — indexed on `txn_id`/`bank`, a `UNIQUE` constraint on the idempotency key, WAL mode for concurrent writers. `/audit`'s JSON shape is unchanged so the dashboard needed no changes. | `execution/db.py` |
| **Request handling under load** | `/recover` hands the pipeline off to a small bounded worker pool (4 workers, 100-deep queue) instead of running unbounded work per request; a saturated pool returns `429` instead of falling over. Throughput and latency under sustained concurrency are measured, not assumed — see [Reliability & Load Testing](#reliability--load-testing) below. | `queueing.py` |
| **Auth** | The control-room login now gets a signed, expiring JWT from `/auth/login` instead of just setting a `localStorage` flag. Off by default (`REQUIRE_AUTH=0`, same zero-config philosophy as the mock LLM) — set `REQUIRE_AUTH=1` to make `/recover` actually verify the token. | `auth.py` |
| **Human-in-the-loop review** | Every `ESCALATE`d decision lands in `/review`. An operator picks the real action and, critically, can correct the *predicted failure reason* — that correction is the labeled training example a retraining pipeline would consume. A model-drift panel tracks the daily override rate (how often operators disagree with the Detective's predicted reason). | `static/review.html`, `execution/db.py` (`pending_review`, `mark_reviewed`, `override_stats`) |
| **Explainability** | Each prediction can be paired with the top contributing features via SHAP (e.g. "predicted `bank_timeout` mainly because: hour=2am, bank=YesBank"), which is the kind of "why did the model say that" answer automated payment decisioning is increasingly expected to give. | `decision/explainability.py` |
| **Tests** | `pytest` suite covering the decision layer's boundary cases (max retries → `STOP`, `card_expired` never retried, amount cap → escalate, low confidence → escalate, mock-LLM branches), idempotent logging, the circuit breaker, and the bandit / EV optimizer / fraud guard / smart routing modules — 53 tests total. Runs on every push via GitHub Actions. | `tests/`, `.github/workflows/test.yml` |
| **Load testing** | `load_test.py` — a dependency-light concurrent load generator for `/recover` reporting p50/p90/p95/p99 latency, throughput, and how often the worker pool's backpressure (`429`) kicks in. | `load_test.py` |

```bash
pip install -r requirements.txt
pytest -q
```

### What I'd do differently at scale

- **Real gateway calls, not simulation** — `_simulate_execution` in `doer.py`
  is the one function that would become a real Razorpay retry API call, an
  SMS/WhatsApp send, or a ticket-system call; the audit contract around it
  wouldn't need to change.
- **Redis/Celery instead of the in-process worker pool** — `queueing.py` is
  intentionally a drop-in seam (`submit(fn, *args)`); at real webhook volume
  I'd swap its implementation for RQ or Celery with a Redis broker so work
  survives a process restart, instead of an in-memory `ThreadPoolExecutor`.
- **Postgres instead of SQLite** — SQLite's single-writer model is fine for a
  demo's write volume; at real scale I'd move to Postgres for proper
  concurrent writers, and add a `bank_id`/`txn_id` composite index sized for
  the actual query patterns instead of the two ad hoc indexes here.
- **A closing half-open state for the circuit breaker** — right now the
  breaker only opens; a production version would move to a half-open state
  after a cooldown, let a small fraction of traffic through as a probe, and
  only fully close once that traffic succeeds — rather than requiring a human
  to manually un-stick a bank.
- **Real accounts behind the review queue** — right now any operator name
  can review a case; a production version would tie `operator` to a real
  logged-in user (via the JWT's `sub` claim) rather than a free-text field.
- **A real cap on escalation volume** — see the honest caveat in
  [Batch Evaluation Results](#batch-evaluation-results): EV mode escalates
  ~45% of cases in this batch because `batch_eval.py` treats human review as
  free and infinite-capacity. A production version would price a queue
  capacity constraint into the EV comparison, not just a per-case cost.

## Why not just "ask an LLM to do everything"?

- LLMs are unreliable at precise probability estimation — that's a job for a
  trained classifier (Detective).
- ML classifiers can't write a context-aware customer message or reason
  through "should this go to a human" — that's a job for an LLM
  (Decision-Maker).
- Neither should be trusted to silently move money without a record — every
  action is logged (Doer), and the riskiest decisions (large amounts, low
  confidence, repeated failures) are governed by rules the LLM cannot
  override.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

python generate_data.py        # 1) synthetic failed-payment dataset -> outputs/
python train_detective.py      # 2) train the two classifiers -> outputs/detective_model.joblib
python batch_eval.py           # 3) run the full pipeline over a batch, print recovery numbers

uvicorn main:app --reload --port 8000
# open http://localhost:8000  ->  the dashboard
```

By default there's no `ANTHROPIC_API_KEY` set, so the Decision-Maker runs on
a deterministic **mock** LLM (clearly labeled `source: mock_llm` in every
response) — the full app works end-to-end with zero API cost. To use the
real Claude-powered decision layer:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload --port 8000
```

`DECISION_MODE` selects which decision layer runs once the hard guardrails,
circuit breaker, and fraud guard have all cleared:

```bash
DECISION_MODE=mock uvicorn main:app --reload      # deterministic rules (default)
DECISION_MODE=llm uvicorn main:app --reload       # real Claude call (needs ANTHROPIC_API_KEY)
DECISION_MODE=ev uvicorn main:app --reload         # expected-value optimizer (real MDR/risk economics)
DECISION_MODE=bandit uvicorn main:app --reload     # self-improving Thompson-Sampling bandit
```

## The 4 things this proves (per the brief)

1. **Money recovered vs baselines** — `python batch_eval.py` runs 200
   synthetic failed payments through the full pipeline and prints recovered
   ₹ vs "do nothing" and "retry everything". Real numbers for both `mock`
   and `ev` decision modes are in
   [Batch Evaluation Results](#batch-evaluation-results) below.
2. **Knows when to stop** — any transaction with `previous_failures >= 3`
   hits the hard guardrail and returns `STOP`; 22 of 200 cases (11%) in the
   batch below hit this.
3. **Knows when to ask a human** — low model confidence or amount above the
   auto-action cap escalates instead of guessing; `batch_eval.py` prints a
   live example.
4. **Every decision is explainable** — `outputs/audit_trail.db` (also
   viewable via `GET /audit` and the dashboard's live ledger) logs the
   diagnosis, the action, the reasoning, and the source for every single
   transaction, optionally paired with SHAP feature contributions via
   `decision/explainability.py`.

## Batch Evaluation Results

`python batch_eval.py` runs 200 synthetic failed payments (₹179,742 total
at risk) through the full pipeline and compares recovered ₹ against two
baselines: "do nothing" (0% by definition) and "retry everything" (blindly
retries every transaction with no cost model, no spam-avoidance, and no
escalation option — not a real strategy, just a ceiling to compare against).

### Mock mode (`DECISION_MODE=mock`, the default)

```
BATCH RESULTS  (n = 200 failed payments)
======================================================================
Total money at risk         : Rs 179,742
Baseline - do nothing       : Rs 0        (0.0% recovered)
Baseline - retry everything : Rs 88,916  (49.5% recovered)
OUR SYSTEM                  : Rs 88,143  (49.0% recovered)

Action breakdown:
  NUDGE_CUSTOMER         51  (25.5%)
  ESCALATE_TO_HUMAN      46  (23.0%)
  RETRY_DELAYED          45  (22.5%)
  RETRY_NOW              36  (18.0%)
  STOP                   22  (11.0%)
```

Mock mode recovers 99.1% of the "retry everything" ceiling (₹88,143 of
₹88,916) — the ~1% gap is the visible cost of the guardrails: 22 cases hit
`STOP` (avoiding spam on customers who've already failed 3+ times) and 46
hit `ESCALATE` (routing low-confidence/high-risk cases to a human instead
of guessing) rather than blindly retrying, which is exactly the tradeoff
those guardrails exist to make.

### Expected-value mode (`DECISION_MODE=ev`)

```
BATCH RESULTS  (n = 200 failed payments)
======================================================================
Total money at risk         : Rs 179,742
Baseline - do nothing       : Rs 0         (0.0% recovered)
Baseline - retry everything : Rs 88,916   (49.5% recovered)
OUR SYSTEM                  : Rs 108,743  (60.5% recovered)

Action breakdown:
  ESCALATE_TO_HUMAN      91  (45.5%)
  RETRY_NOW              44  (22.0%)
  NUDGE_CUSTOMER         24  (12.0%)
  STOP                   22  (11.0%)
  RETRY_DELAYED          19  (9.5%)
```

**EV mode recovers ₹19,827 more than the retry-everything baseline — a 22%
relative improvement** — by pricing every action in rupees (MDR + flat
gateway fee for retries, a risk-flag penalty that scales with prior
failures) and comparing against `HUMAN_RECOVERY_RATE` priors that are often
higher than a bot-only retry for reasons like `bank_timeout` (0.75) or
`network_error` (0.78) — a human reviewer has tools (manual card updates,
direct outreach, risk overrides) a blind retry doesn't.

**Honest caveat:** EV mode escalates 45.5% of cases in this batch — far more
than mock mode's 23%. `batch_eval.py`'s simulation treats human-review
capacity as free and infinite (no queue, no reviewer-hours cap), so this
number is genuinely a **bot+human hybrid vs. bot-only** comparison, not a
bot-vs-bot one. That's arguably the actual pitch (know when to hand off to
a human) rather than a flaw, but a real deployment would need to price a
review-queue capacity constraint into the EV comparison before escalating
at this rate.

### Example: system correctly STOPS instead of endlessly retrying

```
txn_663508: Hit the hard cap of 3 retries. Stopping to avoid spamming the customer.
```

### Example: system ESCALATES to a human instead of guessing (EV mode)

```
txn_395830: [EV] Chose ESCALATE_TO_HUMAN — expected net recovery Rs 1,014.57
(retry cost model: 1.9% MDR + Rs0.5 flat fee, risk-flag penalty Rs0.00) vs
alternatives RETRY_NOW: Rs669, RETRY_DELAYED: Rs669, NUDGE_CUSTOMER: Rs271, STOP: Rs0.
```

Reproduce either run yourself:

```bash
python batch_eval.py                    # mock mode (default)
DECISION_MODE=ev python batch_eval.py   # expected-value mode
```

## Reliability & Load Testing

Two things a payment-recovery agent has to get right before "accuracy" even
matters: it can't lose or double-process a transaction, and it has to hold
up under real concurrency. Both are tested here, not just claimed.

### Idempotency under concurrent duplicate delivery

Failed-payment webhooks in real payment systems are delivered
**at-least-once** — the same event can (and will) be redelivered multiple
times. `execution/doer.py` guards against this with a
`sha256(txn_id + time-bucket)` idempotency key and a `UNIQUE` constraint in
SQLite (`execution/db.py`), so a redelivered webhook returns the *original*
audit row with `duplicate: true` instead of being processed and logged a
second time.

This was verified with `chaos_test_idempotency.py`, which fires the same
`txn_id` at `POST /recover` **50 times concurrently**:

```bash
python chaos_test_idempotency.py --url http://localhost:8000/recover --n 50
```

```json
{
  "total_requests": 50,
  "successful_responses": 50,
  "originals_written": 1,
  "duplicates_detected": 49,
  "errors": 0,
  "PASS": true
}
```

**Result: exactly 1 of 50 concurrent duplicate deliveries was processed and
audited; the other 49 were correctly deduplicated with zero errors.**
Confirmed independently against the database itself:

```bash
sqlite3 outputs/audit_trail.db "SELECT COUNT(*) FROM audit_trail WHERE txn_id='txn_chaos_dup_001';"
# -> 1
```

### Load test — throughput & latency under concurrency

`load_test.py` drives sustained concurrent traffic at `POST /recover` and
reports p50/p95/p99 latency, throughput, and how much traffic the worker
pool's backpressure (`429`) absorbs once it's at capacity — real numbers
instead of an unverified "it's async" claim.

```bash
pip install requests
python load_test.py --url http://localhost:8000/recover --concurrency 50 --duration 30
```

```json
{
  "url": "http://localhost:8000/recover",
  "concurrency": 50,
  "duration_s": 30.8,
  "total_requests": 1862,
  "requests_per_sec": 60.46,
  "status_counts": { "200": 1862 },
  "ok": 1862,
  "throttled_429": 0,
  "errors": 0,
  "latency_ms": {
    "p50": 796.5,
    "p90": 890.6,
    "p95": 920.2,
    "p99": 1294.7,
    "max": 1598.9
  }
}
```

**Result:** 1,862 requests over 30.8s at 50 concurrent workers, **60.5
req/s sustained, zero errors, zero 429s** — every single request succeeded.

**Honest read of the latency numbers:** p50 sitting around ~800ms at only
50 concurrent clients is higher than you'd want for a payments endpoint,
and it's not because the pipeline itself is slow — it's because
`queueing.py`'s worker pool is deliberately bounded to **4 workers**. With
50 concurrent requests arriving against 4 workers actually executing them,
most requests spend the bulk of their time *waiting in the queue*, not
being processed — which is exactly what a bounded pool is supposed to do
(protect the process from unbounded concurrent work) rather than a flaw
that appeared under load. The complete absence of 429s at 50 concurrent
requests means the 100-deep queue never filled — the pool absorbed 12.5x
its own worker count in concurrent load through queuing alone, without
needing to shed any traffic.

**What this number actually says, and what it doesn't:** it proves the
system doesn't fall over, corrupt data, or drop requests under sustained
concurrent load — the reliability claim this section exists to prove. It
is *not* a claim about low-latency performance at this worker count; if
sub-200ms p50 mattered for a real deployment, the fix is exactly what the
"What I'd do differently at scale" section already says — swap the
in-process pool for Redis/Celery with more worker capacity, not a change
to this endpoint's logic.

### Why this matters for a payment system

- **No lost recovery attempts** — a dropped or duplicated webhook can't
  silently cost the merchant a recovered payment or double-charge a
  customer.
- **No silent failures under load** — backpressure (`429`) is explicit and
  measured, not a guess. At 50 concurrent requests the queue absorbed
  every one without shedding traffic.
- **Reproducible, not anecdotal** — both scripts are checked into the repo
  (`chaos_test_idempotency.py`, `load_test.py`) so these numbers can be
  regenerated and verified by anyone reviewing the project, not just quoted
  from memory.

## Project layout

```
main.py                       FastAPI app: /recover, /audit, /auth/login, /outcome/{txn_id},
                               /bandit/arms, /dashboard, /health
queueing.py                   Bounded worker pool /recover submits to
auth.py                       JWT issue/verify for the control-room login
decision/decision_maker.py    Worker 2 — guardrails + circuit breaker + fraud guard +
                               mock/LLM/EV/bandit decision logic (DECISION_MODE)
decision/circuit_breaker.py   Per-bank circuit breaker
decision/smart_routing.py     Suggests an alternate bank when the circuit breaker trips
decision/fraud_guard.py       Card-testing / fraud-cluster detector
decision/ev_optimizer.py      Expected-value action selection priced on real MDR/risk economics
decision/bandit.py            Thompson-Sampling bandit, self-improves from reported outcomes
decision/nudge_copy.py        Reason-aware customer nudge message generation
decision/explainability.py    SHAP-based per-prediction feature explanations
execution/doer.py             Worker 3 — simulated execution + idempotent audit logging
execution/db.py               SQLite-backed audit trail + bandit posterior storage
generate_data.py              Synthetic failed-payment data generator
train_detective.py            Trains Worker 1's two classifiers
batch_eval.py                 Batch evaluation harness (recovery %, examples)
static/dashboard.html         Control-room UI
static/audit.html             Standalone audit ledger UI
static/review.html            Human-in-the-loop review queue UI
tests/                        pytest suite — 53 tests (decision logic, idempotency, circuit
                               breaker, bandit, EV optimizer, fraud guard, smart routing)
load_test.py                  Concurrent load generator for /recover
chaos_test_idempotency.py     Fires duplicate concurrent webhooks to verify idempotency
seed_demo_data.py             Seeds bandit posteriors + circuit-breaker/fraud-guard demo data
                               so those panels aren't empty on a cold demo
.github/workflows/test.yml    CI — runs pytest on every push
Procfile, render.yaml         Deploy config for Render/Railway/Heroku-style hosts
outputs/                      Generated at runtime: dataset, model, audit trail (gitignored)
```

## Deploying

Any host that runs a standard FastAPI/uvicorn app works (Render, Railway,
Fly.io, an EC2 box, etc.). `render.yaml` is set up for a one-click Render
deploy (Blueprint); `Procfile` covers Railway/Heroku-style hosts. Either
way, a host needs to do this before first boot:

```bash
pip install -r requirements.txt
python generate_data.py && python train_detective.py   # produces outputs/detective_model.joblib
```

Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Set `ANTHROPIC_API_KEY` as an environment variable on the host if you want
the real LLM decision layer instead of the mock, and `DECISION_MODE` to
select which decision layer runs (`mock` / `llm` / `ev` / `bandit`).
Optionally set `REQUIRE_AUTH=1` and `JWT_SECRET=<something random>` to
enforce the JWT check on `/recover` (both are off/default in dev so the app
still runs zero-config).

**Note on free-tier hosts:** `outputs/` (the trained model, the SQLite
audit trail, and the bandit's learned posteriors) is regenerated at build
time and typically lives on ephemeral disk — fine for a demo, but this data
will reset on redeploy/restart. For a persistent demo, mount a disk (Render
offers this on paid plans) or point `execution/db.py`'s `DB_PATH` at a
managed Postgres instead. Run `seed_demo_data.py` after each redeploy if
you want the bandit/circuit-breaker/fraud-guard demo panels populated.
