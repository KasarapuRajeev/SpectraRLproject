"""SQLite persistence (stdlib only). One connection per call, guarded by a lock; WAL mode."""
from __future__ import annotations
import json, os, sqlite3, threading, time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.environ.get("SPECTRARL_DATA", Path(__file__).resolve().parents[2] / "data"))
_lock = threading.RLock()
_db_path = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, status TEXT NOT NULL,
  env_json TEXT NOT NULL, ql_json TEXT NOT NULL, baselines INTEGER NOT NULL DEFAULT 1,
  episodes_done INTEGER NOT NULL DEFAULT 0, episodes_total INTEGER NOT NULL,
  created_at REAL NOT NULL, finished_at REAL, error TEXT, summary_json TEXT, policy_path TEXT);
CREATE TABLE IF NOT EXISTS run_metrics(
  run_id INTEGER NOT NULL, episode INTEGER NOT NULL, policy TEXT NOT NULL, eps REAL,
  reward REAL, throughput_mbps REAL, pu_collision REAL, su_collision REAL, utilisation REAL, handovers_per_min REAL,
  PRIMARY KEY(run_id, episode, policy), FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, suite TEXT NOT NULL, status TEXT NOT NULL, quick INTEGER NOT NULL,
  env_json TEXT NOT NULL, ql_json TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, message TEXT,
  created_at REAL NOT NULL, finished_at REAL, error TEXT, result_json TEXT);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id, episode);
"""


def init(path: str | os.PathLike | None = None) -> Path:
    global _db_path
    p = Path(path) if path else DATA_DIR / "spectrarl.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.parent / "policies").mkdir(exist_ok=True)
    _db_path = p
    with conn() as c:
        c.executescript(SCHEMA)
    return p


@contextmanager
def conn():
    with _lock:
        c = sqlite3.connect(_db_path or init(), timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()


def policies_dir() -> Path:
    return (_db_path or init()).parent / "policies"


def row(r):
    return dict(r) if r is not None else None


# ---- runs -----------------------------------------------------------------
def create_run(name, env, ql, baselines) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO runs(name,status,env_json,ql_json,baselines,episodes_total,created_at) VALUES(?,?,?,?,?,?,?)",
                        (name, "queued", env.model_dump_json(), ql.model_dump_json(), int(baselines), ql.episodes, time.time()))
        return cur.lastrowid


def update_run(rid, **kw):
    if not kw:
        return
    with conn() as c:
        c.execute(f"UPDATE runs SET {','.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), rid))


def get_run(rid):
    with conn() as c:
        return row(c.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone())


def list_runs():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC")]


def policy_file(rid):
    """Policy location is derived from the current data dir, so a database copied between machines still works."""
    return policies_dir() / f"run_{rid}.npz"


def delete_run(rid):
    r = get_run(rid)
    with conn() as c:
        c.execute("DELETE FROM run_metrics WHERE run_id=?", (rid,))
        c.execute("DELETE FROM runs WHERE id=?", (rid,))
    if r:
        try:
            os.remove(policy_file(rid))
        except OSError:
            pass


def add_metrics(rid, recs):
    rows = []
    for rec in recs:
        for pol in ("ql", "greedy", "random"):
            if pol in rec:
                m = rec[pol]
                rows.append((rid, rec["episode"], pol, rec["eps"] if pol == "ql" else None, m["reward"], m["throughput_mbps"],
                             m["pu_collision"], m["su_collision"], m["utilisation"], m["handovers_per_min"]))
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO run_metrics VALUES(?,?,?,?,?,?,?,?,?,?)", rows)


def get_metrics(rid, since=0):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM run_metrics WHERE run_id=? AND episode>? ORDER BY episode", (rid, since))]


# ---- experiments ------------------------------------------------------------
def create_experiment(suite, quick, env, ql) -> int:
    with conn() as c:
        return c.execute("INSERT INTO experiments(suite,status,quick,env_json,ql_json,created_at) VALUES(?,?,?,?,?,?)",
                         (suite, "queued", int(quick), env.model_dump_json(), ql.model_dump_json(), time.time())).lastrowid


def update_experiment(eid, **kw):
    if "result_json" in kw and not isinstance(kw["result_json"], str):
        kw["result_json"] = json.dumps(kw["result_json"])
    with conn() as c:
        c.execute(f"UPDATE experiments SET {','.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), eid))


def get_experiment(eid, with_result=True):
    with conn() as c:
        r = row(c.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone())
    if r and with_result and r.get("result_json"):
        r["result"] = json.loads(r["result_json"])
    if r:
        r.pop("result_json", None) if not with_result else None
    return r


def list_experiments():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT id,suite,status,quick,progress,message,created_at,finished_at,error FROM experiments ORDER BY id DESC")]


def latest_experiment(suite, quick=None):
    q = "SELECT id FROM experiments WHERE suite=? AND status='done'" + (" AND quick=?" if quick is not None else "") + " ORDER BY id DESC LIMIT 1"
    with conn() as c:
        r = c.execute(q, (suite, int(quick)) if quick is not None else (suite,)).fetchone()
    return get_experiment(r["id"]) if r else None


def delete_experiment(eid):
    with conn() as c:
        c.execute("DELETE FROM experiments WHERE id=?", (eid,))


def drop_missing_policies():
    """Runs whose policy file is absent (e.g. database copied from another machine) are marked as having no policy."""
    with conn() as c:
        for r in c.execute("SELECT id FROM runs WHERE policy_path IS NOT NULL").fetchall():
            if not policy_file(r["id"]).exists():
                c.execute("UPDATE runs SET policy_path=NULL, status=CASE WHEN status='done' THEN 'failed' ELSE status END, error='policy file not found on this machine' WHERE id=?", (r["id"],))


def mark_interrupted():
    """Jobs that were running when the process died cannot resume; flag them honestly on startup."""
    with conn() as c:
        c.execute("UPDATE runs SET status='interrupted', error='server restarted while running' WHERE status IN ('queued','running')")
        c.execute("UPDATE experiments SET status='interrupted', error='server restarted while running' WHERE status IN ('queued','running')")
