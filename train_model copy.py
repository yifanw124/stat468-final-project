# train_model.py  — Vetiver + pins back-compat (fixed make_s3_board for old pins)
import os, json, inspect
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from xgboost import XGBClassifier
import joblib

# ---------- config ----------
BASE_DIR  = Path("/Users/yifanw124/STAT468/stat468-final-project")
DATA_PATH = BASE_DIR / "tournaments_2018_2025_June.csv"
OUT_DIR   = BASE_DIR
OUT_MODEL = OUT_DIR / "stack_model.joblib"
OUT_META  = OUT_DIR / "feature_spec.json"

PIN_TO_S3          = os.getenv("PIN_TO_S3", "false").lower() == "true"
USE_VETIVER_BUNDLE = os.getenv("USE_VETIVER", "false").lower() == "true"
RANDOM_STATE       = 42

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "")          # used only if PIN_TO_S3
MODEL_PIN    = os.getenv("MODEL_PIN", "stack_model")  # also used as vetiver model_name

# ---------- load ----------
df0 = pd.read_csv(DATA_PATH)
df0 = df0[df0["event"].str.contains("MS|WS", regex=True)].copy()
df0["date"] = pd.to_datetime(df0["date"])
df0 = df0.sort_values("date").reset_index(drop=True)

# ---------- Elo (online, no leakage) ----------
DEFAULT_ELO = 1200
K = 32
elo = defaultdict(lambda: DEFAULT_ELO)

def expected_score(rA, rB):
    return 1 / (1 + 10 ** ((rB - rA) / 400))

def update_elo(rA, rB, outcome_A):
    eA = expected_score(rA, rB)
    rA_new = rA + K * (outcome_A - eA)
    rB_new = rB + K * ((1 - outcome_A) - (1 - eA))
    return rA_new, rB_new

rows = []
for _, r in df0.iterrows():
    p1, p2 = str(r["player1"]), str(r["player2"])
    out1 = 1 if int(r["winner"]) == 1 else 0
    r1, r2 = elo[p1], elo[p2]
    sd = float(r["team1_total_points"] - r["team2_total_points"])

    # features BEFORE updating Elo to avoid leakage
    rows.append({
        "player_id": p1, "opponent_id": p2,
        "elo_player": r1, "elo_opponent": r2,
        "elo_diff": r1 - r2,
        "score_diff": sd,
        "win": out1,
        "date": r["date"],
        "tournament": r.get("tournament_name", None),
        "event": r["event"],
    })
    rows.append({
        "player_id": p2, "opponent_id": p1,
        "elo_player": r2, "elo_opponent": r1,
        "elo_diff": r2 - r1,
        "score_diff": -sd,
        "win": 1 - out1,
        "date": r["date"],
        "tournament": r.get("tournament_name", None),
        "event": r["event"],
    })

    elo[p1], elo[p2] = update_elo(r1, r2, out1)

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ---------- Rolling win% (shifted) ----------
for w in (5, 10, 20):
    df[f"win_pct_{w}"] = (
        df.groupby("player_id")["win"]
          .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    )

# ---------- H2H exponential decay (shifted) ----------
alpha = 0.1
df["h2h_decay"] = (
    df.groupby(["player_id", "opponent_id"])["win"]
      .transform(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean())
)

# Opponent strength adjust (safe divide)
df["h2h_adj"] = (
    df["h2h_decay"] * (df["elo_opponent"] / df["elo_player"].replace(0, np.nan))
).fillna(0.0)

# ---------- Time-based split ----------
date_cut = df["date"].quantile(0.80)
df_tr = df[df["date"] <= date_cut].copy()
df_te = df[df["date"] >  date_cut].copy()

# ---------- PageRank (train period only) ----------
G = nx.DiGraph()
for _, rr in df_tr.iterrows():
    if rr["win"] == 1:
        G.add_edge(rr["opponent_id"], rr["player_id"])
pagerank = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() > 0 else {}

df["pr_player"]   = df["player_id"].map(lambda x: pagerank.get(x, 0.0)).astype(float)
df["pr_opponent"] = df["opponent_id"].map(lambda x: pagerank.get(x, 0.0)).astype(float)

# Re-split after PR
df_tr = df[df["date"] <= date_cut].copy()
df_te = df[df["date"] >  date_cut].copy()

# ---------- Features / target ----------
FEATURES = [
    "elo_diff",
    "win_pct_5", "win_pct_10", "win_pct_20",
    "h2h_decay", "h2h_adj",
    "pr_player", "pr_opponent",
]
for c in FEATURES:
    df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce").fillna(0.0)
    df_te[c] = pd.to_numeric(df_te[c], errors="coerce").fillna(0.0)

X_train, y_train = df_tr[FEATURES], df_tr["win"].astype(int)
X_test,  y_test  = df_te[FEATURES], df_te["win"].astype(int)

# ---------- Model ----------
best_xgb_params = {
    "n_estimators": 214,
    "max_depth": 8,
    "learning_rate": 0.05801866004578234,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}
xgb = XGBClassifier(**best_xgb_params)

estimators = [
    ("lr", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
    ("xgb", xgb),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
    cv=5,
    passthrough=True,
    n_jobs=-1,
)

stack.fit(X_train, y_train)
y_prob = stack.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print(f"[Temporal split] ROC AUC : {roc_auc_score(y_test, y_prob):.6f}")
print(f"[Temporal split] Accuracy: {accuracy_score(y_test, y_pred):.6f}")
print(f"[Temporal split] Brier   : {brier_score_loss(y_test, y_prob):.6f}")

# ---------- Save ----------
players = pd.unique(pd.concat([df["player_id"], df["opponent_id"]], ignore_index=True))
id_to_label = {p: p for p in players}

bundle = {
    "model": stack,
    "features": FEATURES,
    "id_to_label": id_to_label,
    "pagerank": pagerank,
    "trained_on": str(DATA_PATH),
    "date_cutoff": date_cut.isoformat(),
}

if USE_VETIVER_BUNDLE:
    from vetiver import VetiverModel
    import pins, inspect, io, boto3

    # ---- helpers for pins back-compat ----
    def make_s3_board(bucket: str):
        """Use bucket= for new pins, path='s3://bucket' or 'bucket' for old pins."""
        sig = inspect.signature(pins.board_s3)
        if "bucket" in sig.parameters:                 # newer pins API
            return pins.board_s3(bucket=bucket)
        # Older pins API: try with scheme first, then bare
        try:
            return pins.board_s3(path=f"s3://{bucket}")
        except Exception:
            return pins.board_s3(path=bucket)

    def pin_write_compat(board, name, x, type):
        if hasattr(pins, "pin_write"):                 # newer pins module-level
            return pins.pin_write(board, name=name, x=x, type=type)
        if hasattr(board, "pin_write"):                # older pins method
            return board.pin_write(name=name, x=x, type=type)
        raise RuntimeError("pins version not supported: no pin_write found")

    # ---- build Vetiver bundle ----
    v = VetiverModel(model=stack, model_name=MODEL_PIN, prototype=X_train.iloc[:2].copy())
    joblib.dump(v, OUT_DIR / "vetiver_model.joblib")   # optional local artifact

    # ---- primary: try pins to S3 ----
    def try_pins_upload():
        board = make_s3_board(MODEL_BUCKET)
        try:
            pin_write_compat(board, name=MODEL_PIN, x=v, type="vetiver_model")
            print(f"Pinned vetiver model to s3://{MODEL_BUCKET}/{MODEL_PIN} (type=vetiver_model)")
            return True
        except NotImplementedError:
            # older pins can’t handle vetiver_model; try joblib type
            pin_write_compat(board, name=MODEL_PIN, x=v, type="joblib")
            print(f"Pinned VetiverModel via joblib fallback to s3://{MODEL_BUCKET}/{MODEL_PIN}")
            return True

    # ---- fallback: boto3 direct upload (no pins) ----
    def boto3_upload():
        print("pins upload failed; falling back to boto3 direct put to S3…")
        s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        buf = io.BytesIO()
        joblib.dump(v, buf)
        buf.seek(0)
        key = f"{MODEL_PIN}/vetiver_model.joblib"
        s3.put_object(Bucket=MODEL_BUCKET, Key=key, Body=buf.getvalue())
        # optional: write a tiny manifest so API can find it
        manifest = {"type": "vetiver_joblib", "key": key}
        s3.put_object(Bucket=MODEL_BUCKET, Key=f"{MODEL_PIN}/manifest.json",
                      Body=json.dumps(manifest).encode("utf-8"))
        print(f"Uploaded VetiverModel to s3://{MODEL_BUCKET}/{key}")

    if PIN_TO_S3:
        assert MODEL_BUCKET, "MODEL_BUCKET env var must be set when PIN_TO_S3=true"
        try:
            ok = try_pins_upload()
        except Exception as e:
            print(f"pins error: {e!r}")
            ok = False
        if not ok:
            boto3_upload()
else:
    # Plain joblib bundle for a minimal FastAPI (not used by Vetiver API)
    joblib.dump(bundle, OUT_MODEL)
    with open(OUT_META, "w") as f:
        json.dump({"features": FEATURES, "types": {c: "float" for c in FEATURES}}, f, indent=2)
    print(f"Saved model bundle to {OUT_MODEL}")
