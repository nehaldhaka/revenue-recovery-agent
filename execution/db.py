"""
execution/db.py
----------------
SQLite-backed audit trail. Replaces the original CSV file so we get:

  - real write-safety when requests happen concurrently (WAL mode +
    a write lock, instead of N threads appending to the same file
    handle),
  - an index on txn_id/bank so the circuit breaker's "recent
    transactions for this bank" query and any future txn_id lookup
    are O(log n) instead of a full file scan,
  - a natural place to enforce idempotency (UNIQUE constraint on
    idempotency_key) instead of re-reading the whole file on every
    write to check for duplicates.

`/audit` still returns the exact same JSON shape the dashboard/audit
UIs already expect, so nothing on the frontend has to change.

NEW in this version:
  - an `outcome` column on audit_trail ('recovered' | 'failed' | NULL)
    recorded via POST /outcome/{txn_id} once we actually know whether
    a retry/nudge/escalation worked. This is the ground truth both
    the bandit (decision/bandit.py) and smart routing
    (decision/smart_routing.py) learn from.
  - a bandit_arms table storing the Beta(alpha, beta) posterior for
    each (action, reason, amount_bucket) arm the Thompson-Sampling
    bandit maintains.
  - recent_rows() and pending_review() now exclude source='seed_script'
    rows: those are synthetic data from seed_demo_data.py, meant only
    to populate the circuit-breaker / fraud-guard / smart-routing
    *detection logic* (which reads recent_for_bank() separately,
    unaffected by this filter) — they were never meant to show up as
    if they were real transactions in the Audit Trail or Review Queue
    a user is actually looking at.
"""
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

DB_PATH = os.path.join("outputs", "audit_trail.db")

_local = threading.local()
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_trail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT UNIQUE,
    logged_at TEXT NOT NULL,
    txn_id TEXT NOT NULL,
    amount REAL,
    bank TEXT,
    hour INTEGER,
    previous_failures INTEGER,
    is_subscription INTEGER,
    days_since_last_success INTEGER,
    predicted_reason TEXT,
    confidence REAL,
    retry_success_prob REAL,
    action TEXT,
    wait_hours INTEGER,
    customer_message TEXT,
    reasoning TEXT,
    source TEXT,
    execution_note TEXT,
    duplicate_of INTEGER,
    reviewed INTEGER DEFAULT 0,
    operator TEXT,
    operator_action TEXT,
    corrected_reason TEXT,
    reviewed_at TEXT,
    outcome TEXT,
    outcome_logged_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_txn_id ON audit_trail(txn_id);
CREATE INDEX IF NOT EXISTS idx_audit_bank ON audit_trail(bank);
CREATE INDEX IF NOT EXISTS idx_audit_logged_at ON audit_trail(logged_at);

CREATE TABLE IF NOT EXISTS bandit_arms (
    arm_key TEXT PRIMARY KEY,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    pulls INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Adds columns from later versions of this schema to a DB created
    before they existed, so upgrading doesn't require deleting
    outputs/audit_trail.db."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(audit_trail)").fetchall()}
    for col, decl in [
        ("reviewed", "INTEGER DEFAULT 0"),
        ("operator", "TEXT"),
        ("operator_action", "TEXT"),
        ("corrected_reason", "TEXT"),
        ("reviewed_at", "TEXT"),
        ("outcome", "TEXT"),
        ("outcome_logged_at", "TEXT"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE audit_trail ADD COLUMN {col} {decl}")
    conn.commit()


def get_conn() -> sqlite3.Connection:
    """One connection per thread (SQLite connections aren't safe to
    share across threads); the worker pool in queueing.py means we
    genuinely have several threads hitting this at once."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA)
        _migrate(conn)
        _local.conn = conn
    return conn


def find_by_idempotency_key(key: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM audit_trail WHERE idempotency_key = ?", (key,)
    ).fetchone()
    return dict(row) if row else None


def insert_row(row: dict) -> dict:
    conn = get_conn()
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    with _write_lock:
        cur = conn.execute(
            f"INSERT INTO audit_trail ({','.join(cols)}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )
        conn.commit()
        row_id = cur.lastrowid
    saved = dict(row)
    saved["id"] = row_id
    return saved


def recent_rows(limit: int = 30):
    """Newest first — matches the ordering the original CSV endpoint
    produced (it reversed the last N lines of the file).

    Excludes source='seed_script' rows so the Audit Trail only shows
    real payments submitted through Control Room's "New failed
    payment" form, not the synthetic demo data seed_demo_data.py
    writes to trip the circuit breaker / fraud guard / smart routing."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM audit_trail WHERE source != 'seed_script' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def recent_for_bank(bank: str, since_iso: str):
    """Deliberately NOT filtered by source — circuit_breaker.py and
    fraud_guard.py need to see seed_script rows for the demo triggers
    to keep working. This function feeds detection logic, not a
    user-facing display."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM audit_trail WHERE bank = ? AND logged_at >= ? ORDER BY id DESC",
        (bank, since_iso),
    ).fetchall()
    return [dict(r) for r in rows]


def find_latest_by_txn_id(txn_id: str):
    """Most recent audit row for a given txn_id — used by
    POST /outcome/{txn_id} to look up which action/reason/amount an
    outcome report should be attributed to."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM audit_trail WHERE txn_id = ? ORDER BY id DESC LIMIT 1",
        (txn_id,),
    ).fetchone()
    return dict(row) if row else None


def record_outcome(txn_id: str, recovered: bool):
    """Records the REAL outcome of a case after the fact (did the
    retry/nudge/escalation actually recover the payment). This is the
    ground-truth signal the bandit and smart-routing modules learn
    from — without it they'd just be guessing forever."""
    row = find_latest_by_txn_id(txn_id)
    if row is None:
        return None

    outcome = "recovered" if recovered else "failed"
    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    with _write_lock:
        conn.execute(
            "UPDATE audit_trail SET outcome = ?, outcome_logged_at = ? WHERE id = ?",
            (outcome, now, row["id"]),
        )
        conn.commit()

    row["outcome"] = outcome
    row["outcome_logged_at"] = now
    return row


def pending_review(limit: int = 50):
    """Escalated decisions an operator hasn't weighed in on yet,
    oldest first (so the queue drains in order).

    Excludes source='seed_script' for the same reason as recent_rows()
    above — seeded fraud/circuit-breaker demo cases shouldn't show up
    as real work waiting on a human operator."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM audit_trail "
        "WHERE action = 'ESCALATE' AND (reviewed IS NULL OR reviewed = 0) "
        "AND source != 'seed_script' "
        "ORDER BY id ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_reviewed(record_id: int, operator: str, action: str, corrected_reason: str = None):
    """Records an operator's decision on an escalated case. Returns
    the updated row, or None if record_id doesn't exist."""
    conn = get_conn()
    existing = conn.execute("SELECT * FROM audit_trail WHERE id = ?", (record_id,)).fetchone()
    if existing is None:
        return None

    reviewed_at = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn.execute(
            "UPDATE audit_trail "
            "SET reviewed = 1, operator = ?, operator_action = ?, "
            "corrected_reason = ?, reviewed_at = ? WHERE id = ?",
            (operator, action, corrected_reason, reviewed_at, record_id),
        )
        conn.commit()

    row = dict(existing)
    row.update(
        reviewed=1,
        operator=operator,
        operator_action=action,
        corrected_reason=corrected_reason,
        reviewed_at=reviewed_at,
    )
    return row


def override_stats():
    """
    Daily reviewed-vs-overridden counts. An "override" here means the
    operator corrected the predicted failure reason — i.e. the
    Detective model was wrong, not just that a human made the final
    call (which is expected for every escalated case). A rising
    override_rate over time is the model-drift signal: the Detective
    is getting the *reason* wrong more often, which is worth
    retraining on.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT logged_at, predicted_reason, corrected_reason FROM audit_trail "
        "WHERE reviewed = 1 ORDER BY logged_at"
    ).fetchall()

    buckets = {}
    for r in rows:
        day = (r["logged_at"] or "")[:10]
        b = buckets.setdefault(day, {"reviewed": 0, "overridden": 0})
        b["reviewed"] += 1
        if r["corrected_reason"] and r["corrected_reason"] != r["predicted_reason"]:
            b["overridden"] += 1

    result = []
    for day in sorted(buckets):
        b = buckets[day]
        rate = (b["overridden"] / b["reviewed"]) if b["reviewed"] else 0.0
        result.append({"day": day, "reviewed": b["reviewed"], "overridden": b["overridden"], "override_rate": rate})
    return result


# ============================================================
# retrain.py's feedback loop
# ============================================================

def human_reviewed_rows():
    """
    Every case an operator has finished reviewing, with the corrected
    (ground-truth) failure reason. This is the labeled training data
    retrain.py merges into the Detective's next training run — the
    operator's `corrected_reason` is what actually happened, not what
    the model guessed.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT txn_id, amount, bank, hour, previous_failures, is_subscription, "
        "days_since_last_success, predicted_reason, corrected_reason, reviewed_at "
        "FROM audit_trail "
        "WHERE reviewed = 1 AND corrected_reason IS NOT NULL AND corrected_reason != ''"
    ).fetchall()
    return [dict(r) for r in rows]


def rolling_override_rate(window_days: int = 7):
    """
    Same override-rate signal as override_stats(), but as a single
    rolling number over the last `window_days` (keyed off reviewed_at,
    i.e. when the human actually made the call) instead of a
    day-bucketed history. This is what the auto-retrain trigger checks
    against a threshold (see retrain.py: should_retrain()).
    """
    conn = get_conn()
    since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    rows = conn.execute(
        "SELECT predicted_reason, corrected_reason FROM audit_trail "
        "WHERE reviewed = 1 AND reviewed_at >= ?",
        (since,),
    ).fetchall()

    reviewed = len(rows)
    overridden = sum(
        1 for r in rows if r["corrected_reason"] and r["corrected_reason"] != r["predicted_reason"]
    )
    rate = (overridden / reviewed) if reviewed else 0.0

    return {
        "window_days": window_days,
        "reviewed": reviewed,
        "overridden": overridden,
        "override_rate": rate,
    }


# ============================================================
# NEW — bandit arm persistence (decision/bandit.py)
# ============================================================

def get_bandit_arm(arm_key: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM bandit_arms WHERE arm_key = ?", (arm_key,)
    ).fetchone()
    return dict(row) if row else None


def upsert_bandit_arm(arm_key: str, alpha: float, beta: float, pulls: int):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn.execute(
            "INSERT INTO bandit_arms (arm_key, alpha, beta, pulls, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(arm_key) DO UPDATE SET alpha=excluded.alpha, "
            "beta=excluded.beta, pulls=excluded.pulls, updated_at=excluded.updated_at",
            (arm_key, alpha, beta, pulls, now),
        )
        conn.commit()


def all_bandit_arms():
    """Every arm's current posterior — useful for a debug/inspection
    endpoint or a dashboard panel showing what the bandit has learned."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM bandit_arms ORDER BY pulls DESC").fetchall()
    return [dict(r) for r in rows]