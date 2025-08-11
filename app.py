# app.py
from __future__ import annotations
import io, os, pickle, warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

# boto3 for S3
try:
    import boto3
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False

from fastapi import FastAPI, HTTPException
from vetiver import VetiverModel, VetiverAPI
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from shiny import App, reactive, render, ui

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# S3 defaults
# ============================================================
DEFAULT_BUCKET = os.getenv("MODEL_BUCKET", "badminton12345")
DEFAULT_MODEL_KEY = os.getenv("MODEL_KEY", "stack_model.pkl")
DEFAULT_DATA_KEY  = os.getenv("DATA_KEY",  "tournaments_2018_2025_June.csv")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

HERE = Path(__file__).resolve().parent
LOCAL_MODEL = HERE / "stack_model.pkl"

# ============================================================
# Helpers
# ============================================================
def s3_client():
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed. pip install boto3")
    return boto3.client("s3", region_name=AWS_REGION)

def load_pickle_model() -> Any:
    """Load model from S3 (bucket/key above) or local fallback."""
    if HAS_BOTO3:
        try:
            obj = s3_client().get_object(Bucket=DEFAULT_BUCKET, Key=DEFAULT_MODEL_KEY)
            return pickle.load(io.BytesIO(obj["Body"].read()))
        except Exception as e:
            print(f"[warn] Could not load model from s3://{DEFAULT_BUCKET}/{DEFAULT_MODEL_KEY}: {e}")
    if LOCAL_MODEL.exists():
        with open(LOCAL_MODEL, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"Model not found in S3 or locally. "
        f"Tried s3://{DEFAULT_BUCKET}/{DEFAULT_MODEL_KEY} and {LOCAL_MODEL}"
    )

def find_estimator(x: Any) -> Any:
    if hasattr(x, "predict"):
        return x
    if isinstance(x, dict):
        for v in x.values(): 
            m = find_estimator(v)
            if m is not None: return m
    if isinstance(x, (list, tuple)):
        for v in x: 
            m = find_estimator(v)
            if m is not None: return m
    return None

# ============================================================
# Load model (dict or raw estimator)
# ============================================================
SAVED = load_pickle_model()

if isinstance(SAVED, dict):
    MODEL = SAVED.get("model") or find_estimator(SAVED)
    FEATURES: List[str] = list(SAVED.get("features") or [])
    ID_TO_LABEL: Dict[str,str] = dict(SAVED.get("id_to_label") or {})
    PRECOMP_PR = SAVED.get("pagerank")
else:
    MODEL = find_estimator(SAVED)
    FEATURES = []
    ID_TO_LABEL = {}
    PRECOMP_PR = None

if MODEL is None:
    raise RuntimeError("Could not find estimator with .predict() inside the pickle.")

DEFAULT_FEATURES = [
    "elo_diff","win_pct_5","win_pct_10","win_pct_20",
    "h2h_decay","h2h_adj","pr_player","pr_opponent"
]
if not FEATURES: FEATURES = DEFAULT_FEATURES[:]
X_PROTO = pd.DataFrame([{c: 0.0 for c in FEATURES}]).astype({c:"float64" for c in FEATURES})

# ============================================================
# Feature engineering for uploaded (or default) CSVs
# ============================================================
DEFAULT_ELO = 1200
K = 32
REQUIRED_COLS = [
    "date","event","player1","player2","winner",
    "team1_total_points","team2_total_points","tournament_name"
]

def _expected_score(rA, rB): return 1 / (1 + 10 ** ((rB - rA) / 400))
def _update_elo(rA, rB, outA):
    eA = _expected_score(rA, rB)
    return rA + K*(outA - eA), rB + K*((1-outA) - (1-eA))

def build_features_from_raw(df_raw: pd.DataFrame, precomp_pr: Dict[str,float] | None=None) -> pd.DataFrame:
    miss = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if miss: raise ValueError(f"Missing required columns: {miss}")

    df = df_raw.copy()
    df = df[df["event"].str.contains("MS|WS", regex=True)]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    elo = defaultdict(lambda: DEFAULT_ELO)
    rows = []
    for _, r in df.iterrows():
        p1, p2 = r["player1"], r["player2"]
        out1 = 1 if r["winner"] == 1 else 0
        r1, r2 = elo[p1], elo[p2]
        sd = r["team1_total_points"] - r["team2_total_points"]

        rows.append({"player_id":p1,"opponent_id":p2,"elo_player":r1,"elo_opponent":r2,
                     "elo_diff":r1-r2,"score_diff":sd,"win":out1,"date":r["date"],
                     "tournament":r["tournament_name"],"event":r["event"]})
        rows.append({"player_id":p2,"opponent_id":p1,"elo_player":r2,"elo_opponent":r1,
                     "elo_diff":r2-r1,"score_diff":-sd,"win":1-out1,"date":r["date"],
                     "tournament":r["tournament_name"],"event":r["event"]})
        elo[p1], elo[p2] = _update_elo(r1, r2, out1)

    df_feat = pd.DataFrame(rows)

    for w in (5,10,20):
        df_feat[f"win_pct_{w}"] = (
            df_feat.sort_values("date")
                  .groupby("player_id", group_keys=False)["win"]
                  .apply(lambda s: s.rolling(w, min_periods=1).mean())
        )

    decay = 0.9
    def decayed(seq):
        v=0; out=[]
        for x in seq: v=decay*v+x; out.append(v)
        return out
    df_feat = df_feat.sort_values("date")
    df_feat["h2h_decay"] = (
        df_feat.groupby(["player_id","opponent_id"], group_keys=False)["win"]
               .apply(decayed)
    )
    df_feat["h2h_adj"] = df_feat["h2h_decay"] * (df_feat["elo_opponent"] / (df_feat["elo_player"] + 1e-6))

    if precomp_pr is None:
        G = nx.DiGraph()
        for _, r in df_feat.iterrows():
            if r["win"] == 1: G.add_edge(r["opponent_id"], r["player_id"])
        pr = nx.pagerank(G, alpha=0.85)
    else:
        pr = {str(k): float(v) for k,v in precomp_pr.items()}

    df_feat["pr_player"]   = df_feat["player_id"].map(lambda x: pr.get(str(x),0.0)).fillna(0.0)
    df_feat["pr_opponent"] = df_feat["opponent_id"].map(lambda x: pr.get(str(x),0.0)).fillna(0.0)

    for c in FEATURES:
        if c not in df_feat.columns: df_feat[c] = 0.0
    df_feat[FEATURES] = df_feat[FEATURES].fillna(0)
    return df_feat[["player_id","opponent_id","date"] + FEATURES + ["win"]]

def load_default_csv_from_s3() -> pd.DataFrame | None:
    """Used as a default dataset if user doesn't upload."""
    if not HAS_BOTO3: return None
    try:
        obj = s3_client().get_object(Bucket=DEFAULT_BUCKET, Key=DEFAULT_DATA_KEY)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        print(f"[warn] Could not load default CSV from s3://{DEFAULT_BUCKET}/{DEFAULT_DATA_KEY}: {e}")
        return None

DEFAULT_RAW = load_default_csv_from_s3()

# ============================================================
# FastAPI + Vetiver (api)
# ============================================================
api = FastAPI(title="Stack Model API + Vetiver")

@api.get("/health")
def health():
    return {"status":"ok","model_type":type(MODEL).__name__,"features":FEATURES}

@api.post("/predict")
def predict(payload: Dict[str, list]):
    try:
        rows = payload.get("data", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("Payload must include non-empty 'data' list")
        X = pd.DataFrame(rows)
        missing = [c for c in FEATURES if c not in X.columns]
        if missing: raise ValueError(f"Missing features: {missing}")
        X = X[FEATURES]
        preds = MODEL.predict(X).tolist()
        out = {"predictions": preds}
        if hasattr(MODEL, "predict_proba"):
            out["predict_proba"] = MODEL.predict_proba(X).tolist()
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

VetiverAPI(VetiverModel(MODEL, "stack-model", prototype_data=X_PROTO), app=api, check_prototype=True)

# ============================================================
# Py-Shiny app (app)
# ============================================================
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Upload tournaments CSV", accept=[".csv"], multiple=False),
        ui.input_action_button("go", "Run predictions"),
        ui.markdown(
            f"Default dataset: `s3://{DEFAULT_BUCKET}/{DEFAULT_DATA_KEY}` (used only if no file is uploaded)."
        ),
        ui.input_checkbox("show_eval", "If upload/default includes `winner`, compute quick metrics", True),
    ),
    ui.h3("Badminton Win Probability"),
    ui.card(ui.card_header("Predictions"), ui.output_table("pred_table")),
    ui.card(ui.card_header("Metrics (optional)"), ui.output_text("metrics")),
    title="Badminton H2H Predictor",
)

def shiny_server(input, output, session):

    @reactive.calc
    def raw_df():
        f = input.csv()
        if f:
            return pd.read_csv(f[0]["datapath"])
        return DEFAULT_RAW  # may be None if S3 read failed

    @reactive.calc
    def features_df():
        df_raw = raw_df()
        if df_raw is None:
            return None
        return build_features_from_raw(df_raw, precomp_pr=PRECOMP_PR)

    @reactive.event(input.go)
    def _run():
        pass

    @output
    @render.table
    def pred_table():
        _ = _run()
        feats = features_df()
        if feats is None or feats.empty:
            return pd.DataFrame()
        X = feats[FEATURES]
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[:, 1]
        else:
            proba = MODEL.predict(X)
        out = feats[["player_id","opponent_id","date"]].copy()
        out["win_prob_player_id"] = np.asarray(proba).ravel()
        return out.sort_values("date")

    @output
    @render.text
    def metrics():
        if not input.show_eval(): return "(hidden)"
        feats = features_df()
        if feats is None or "win" not in feats.columns:
            return "No labels present; metrics skipped."
        try:
            X = feats[FEATURES]
            y = feats["win"].astype(int)
            if hasattr(MODEL, "predict_proba"):
                p = MODEL.predict_proba(X)[:, 1]
            else:
                p = MODEL.predict(X)
            acc = accuracy_score(y, (np.asarray(p) >= 0.5).astype(int))
            return f"AUC={roc_auc_score(y,p):.3f} | Brier={brier_score_loss(y,p):.3f} | Acc={acc:.3f}"
        except Exception as e:
            return f"Metrics unavailable: {e}"

app = App(app_ui, shiny_server)
