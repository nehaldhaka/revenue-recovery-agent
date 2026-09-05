# Defense Notes

Study tool, not a script. Each section is "the question a judge would
actually ask" + "the honest reasoning," so you can explain it in your
own words instead of reciting mine. If you read a "why" and it doesn't
click, that's the thing to actually re-derive before your interview —
not paper over.

---

## `decision/bandit.py` — Thompson Sampling

**Q: Why Beta distribution specifically, and not just track a raw success rate?**

A raw success rate (successes/attempts) treats "1 win out of 1 try" the
same confidence-wise as "500 wins out of 500 tries" — both say 100%.
That's wrong: the first is almost pure luck, the second is real
evidence. The Beta distribution's *shape* encodes that uncertainty —
Beta(2,1) is wide and uncertain, Beta(501,1) is a narrow spike near 1.0.
Sampling from the distribution (not just reading its mean) means an
arm with little data still gets explored sometimes, because its wide
Beta occasionally samples a high value by chance. That's the
exploration mechanism — no separate epsilon-greedy schedule needed.

**Q: Why does alpha go up on success and beta on failure — where does that come from?**

Beta(alpha, beta) is the *conjugate prior* for a Bernoulli/Binomial
likelihood — meaning if your prior belief is Beta(a,b) and you observe
one more Bernoulli trial, the posterior is exactly Beta(a+1, b) on
success or Beta(a, b+1) on failure. It's not a design choice you
picked, it's the mathematically exact Bayesian update for this
distribution pair. This is why `record_outcome` is just `alpha += 1`
or `beta += 1` — the math is that simple *because* Beta-Bernoulli was
chosen as the model in the first place.

**Q: Why start every arm at Beta(1,1) instead of Beta(0,0)?**

Beta(1,1) is the uniform distribution on [0,1] — "I have literally no
opinion yet, every success rate from 0% to 100% is equally likely."
Beta(0,0) isn't a valid probability distribution (it's improper —
doesn't integrate to 1), so Beta(1,1) is the standard "maximally
uninformative but valid" starting point.

**Q: Why bucket by amount instead of using the exact rupee value?**

An arm is `(action, reason, amount)`. If amount were exact, virtually
every transaction would create a brand-new arm with zero history, and
the bandit could never learn anything — you'd need millions of ₹4,317
transactions before that exact arm had any data. Bucketing (`xs/s/m/l/xl`)
trades precision for enough shared history per arm to actually learn
within a demo's data volume.

**Q: What's the actual failure mode if you seeded fake outcomes to make this "look smart" for judges — is that dishonest?**

Be upfront about it if asked: `seed_demo_data.py` exists explicitly to
avoid a cold, empty demo (empty is what a fresh Beta(1,1) prior looks
like everywhere), and the docstring says so. The honest answer under
questioning is "the algorithm is real, the demo scenario is synthetic to
show it working — the same pattern any ML demo uses with synthetic
training data."

---

## `decision/ev_optimizer.py` — Cost-priced decisioning

**Q: Where do MDR_RATE (1.9%) and the other constants actually come from?**

Be honest: they're reasonable placeholder assumptions, not numbers
pulled from a real Razorpay pricing sheet. The point isn't "these are
Razorpay's actual numbers" — it's "the *architecture* separates real
cost inputs from the decision logic, so swapping in real finance/risk
numbers later is a one-line constant change, not a rewrite." That's
the defensible claim.

**Q: Why does the risk-flag penalty scale with `previous_failures` specifically?**

Card networks and issuers watch for unusual retry volume on the same
card as one of several fraud signals — a card that's failed 3 times
and gets hit with another retry attempt looks more like probing than
one that's failing for the first time. Scaling the penalty by prior
failures encodes "the more we've already hammered this transaction,
the more suspicious one more attempt looks" — which is directionally
true even if the exact multiplier (0.04 per failure) is a guess.

**Q: RETRY_DELAYED gets only half the risk penalty of RETRY_NOW — why?**

The risk pattern networks flag is rapid repeated attempts. Spacing a
retry out (delayed vs. immediate) is specifically the thing that makes
it look less like automated probing and more like a genuine follow-up
attempt — so it's modeled as carrying less of that penalty. This is a
modeling assumption, own it as one if asked to justify the "0.5"
specifically — there's no first-principles reason it isn't 0.4 or 0.6.

---

## `decision/fraud_guard.py` — Card-testing detector

**Q: Why "many low-value transactions in a tight amount band" specifically, as the fraud signature?**

Card-testing (validating stolen card numbers work) needs the *smallest*
possible charge to minimize detection risk to the attacker before they
know if the card is live — so real card-testing traffic clusters at
low, similar amounts, fired in bursts. A genuine customer's failed
payments, by contrast, are naturally varied in amount (whatever they
were actually trying to buy) and spread out in time. The tight
low-amount cluster *is* the distinguishing signal, not an arbitrary
threshold choice.

**Q: Why measure `stdev / mean` (coefficient of variation) instead of just checking if all amounts are within some fixed ₹X of each other?**

A fixed absolute band (e.g., "within ₹5 of each other") breaks at
different amount scales — ₹5 tight at ₹10 mean, but that same ₹5
window is nothing at a ₹90 mean. `stdev/mean` is scale-independent: it
answers "how tight is this cluster *relative to its own size*," which
is the actual property that matters regardless of what the average
amount happens to be.

---

## `decision/smart_routing.py` — Alternate-bank suggestion

**Q: Why is this "advisory only" — why not have it actually reroute the transaction?**

Two honest reasons: (1) scope — this demo has no real multi-PSP
integration, so "recommend" is what's actually implementable without
faking a capability that doesn't exist; (2) it's the more defensible
design anyway — a human or a higher-level system should confirm a
routing change with real business context (contractual routing rules,
compliance constraints) that this module doesn't have visibility into.

**Q: Why require `MIN_ROUTE_SAMPLES = 3` before recommending anything?**

Recommending based on 1 sample (e.g., "HDFC recovered its one recent
case, so 100%!") is statistically meaningless — it's exactly the
small-sample overconfidence problem the Beta-distribution reasoning
above addresses for the bandit. Requiring a minimum sample size before
speaking up is the cheap, simple version of the same idea.

---

## `decision/explainability.py` — SHAP

**Q: Why SHAP specifically, and why TreeExplainer?**

SHAP values answer "how much did each feature push *this specific
prediction* away from the model's average prediction" — grounded in
game theory (Shapley values from cooperative game theory), which gives
it a rigorous, consistent definition of "contribution" rather than an
ad-hoc heuristic. `TreeExplainer` is the fast, exact variant built
specifically for tree-based models (your RandomForest) — a generic
model-agnostic SHAP explainer would be much slower and only
approximate for the same model.

**Q: Why wrap it in try/except and return `[]` on failure?**

Explainability is an enrichment, not a dependency — a prediction must
never fail or block because an optional interpretability library isn't
installed or throws on an edge case. This is a defensible engineering
choice to name explicitly if asked: "critical path vs. nice-to-have"
separation.

---

## `auth.py` — JWT, off by default

**Q: Why off by default (`REQUIRE_AUTH=0`)? Isn't that insecure?**

For a demo/buildathon context, forcing auth configuration before the
app even runs would mean judges can't try it zero-config — that's a
worse failure mode than a demo running unauthenticated. The same
pattern as the mock LLM: "works out of the box, hardens when you flip
one env var." Own the tradeoff explicitly rather than pretending it's
production-secure by default.

---

## General framing for "how does this scale to production"

If asked what you'd change for real production load, the honest answer
is already written down in the README's "What I'd do differently at
scale" section — Redis/Celery instead of the in-process worker pool,
Postgres instead of SQLite, a half-open circuit-breaker state instead
of manual-only recovery, real accounts instead of free-text operator
names. Knowing that list cold — and *why* each swap matters — is worth
more in an interview than any single line of code.