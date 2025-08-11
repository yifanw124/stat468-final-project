from __future__ import annotations
import io, os, pickle, warnings, json
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import requests
from fastapi import FastAPI, HTTPException
from vetiver import VetiverModel, VetiverAPI
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from shiny import App, reactive, render, ui
from plotnine import ggplot, aes, geom_histogram, geom_density, geom_point, geom_smooth, facet_wrap, labs, theme_bw
from great_tables import GT, style, loc
from htmltools import HTML

warnings.filterwarnings("ignore", category=UserWarning)

# config
DEFAULT_BUCKET = os.getenv("MODEL_BUCKET", "badminton12345")
DEFAULT_MODEL_KEY = os.getenv("MODEL_KEY", "stack_model.pkl")
DEFAULT_DATA_KEY  = os.getenv("DATA_KEY",  "tournaments_2018_2025_June.csv")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")
HERE = Path(__file__).resolve().parent
LOCAL_MODEL = HERE / "stack_model.pkl"

# s3
try:
    import boto3
    def s3_client(): return boto3.client("s3", region_name=AWS_REGION)
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False

# load
def load_pickle_model() -> Any:
    if HAS_BOTO3:
        try:
            obj = s3_client().get_object(Bucket=DEFAULT_BUCKET, Key=DEFAULT_MODEL_KEY)
            return pickle.load(io.BytesIO(obj["Body"].read()))
        except Exception as e:
            print(f"[warn] s3 model {DEFAULT_BUCKET}/{DEFAULT_MODEL_KEY}: {e}")
    if LOCAL_MODEL.exists():
        with open(LOCAL_MODEL, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("no model found in S3 or local")

def find_estimator(x: Any) -> Any:
    if hasattr(x, "predict"): return x
    if isinstance(x, dict):
        for v in x.values():
            m = find_estimator(v)
            if m is not None: return m
    if isinstance(x, (list, tuple)):
        for v in x:
            m = find_estimator(v)
            if m is not None: return m
    return None

SAVED = load_pickle_model()
if isinstance(SAVED, dict):
    MODEL = SAVED.get("model") or find_estimator(SAVED)
    FEATURES: List[str] = list(SAVED.get("features") or [])
    ID_TO_LABEL: Dict[str,str] = dict(SAVED.get("id_to_label") or {})
    PRECOMP_PR = SAVED.get("pagerank")
else:
    MODEL = find_estimator(SAVED); FEATURES=[]; ID_TO_LABEL={}; PRECOMP_PR=None
if MODEL is None: raise RuntimeError("no estimator with .predict() in pickle")

DEFAULT_FEATURES = ["elo_diff","win_pct_5","win_pct_10","win_pct_20","h2h_decay","h2h_adj","pr_player","pr_opponent"]
if not FEATURES: FEATURES = DEFAULT_FEATURES[:]
X_PROTO = pd.DataFrame([{c: 0.0 for c in FEATURES}]).astype({c:"float64" for c in FEATURES})

# features
DEFAULT_ELO = 1200
K = 32
REQUIRED_COLS = ["date","event","player1","player2","winner","team1_total_points","team2_total_points","tournament_name"]

def _expected_score(rA, rB): return 1 / (1 + 10 ** ((rB - rA) / 400))
def _update_elo(rA, rB, outA):
    eA = _expected_score(rA, rB)
    return rA + K*(outA - eA), rB + K*((1-outA) - (1-eA))

def build_features_from_raw(df_raw: pd.DataFrame, precomp_pr: Dict[str,float] | None=None) -> pd.DataFrame:
    miss = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if miss: raise ValueError(f"missing cols: {miss}")
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
        rows.append({"player_id":p1,"opponent_id":p2,"elo_player":r1,"elo_opponent":r2,"elo_diff":r1-r2,"score_diff":sd,"win":out1,"date":r["date"],"tournament":r["tournament_name"],"event":r["event"]})
        rows.append({"player_id":p2,"opponent_id":p1,"elo_player":r2,"elo_opponent":r1,"elo_diff":r2-r1,"score_diff":-sd,"win":1-out1,"date":r["date"],"tournament":r["tournament_name"],"event":r["event"]})
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
    df_feat["h2h_decay"] = df_feat.groupby(["player_id","opponent_id"], group_keys=False)["win"].apply(decayed)
    df_feat["h2h_adj"] = df_feat["h2h_decay"] * (df_feat["elo_opponent"] / (df_feat["elo_player"] + 1e-6))
    if precomp_pr is None:
        G = nx.DiGraph()
        for _, r in df_feat.iterrows():
            if r["win"] == 1: G.add_edge(r["opponent_id"], r["player_id"])
        pr = nx.pagerank(G, alpha=0.85)
    else:
        pr = {str(k): float(v) for k,v in (precomp_pr or {}).items()}
    df_feat["pr_player"]   = df_feat["player_id"].map(lambda x: pr.get(str(x),0.0)).fillna(0.0)
    df_feat["pr_opponent"] = df_feat["opponent_id"].map(lambda x: pr.get(str(x),0.0)).fillna(0.0)
    for c in FEATURES:
        if c not in df_feat.columns: df_feat[c] = 0.0
    df_feat[FEATURES] = df_feat[FEATURES].fillna(0)
    keep = ["player_id","opponent_id","date","event"] + FEATURES + ["win"]
    return df_feat[keep]

def load_default_csv_from_s3() -> pd.DataFrame | None:
    if not HAS_BOTO3: return None
    try:
        obj = s3_client().get_object(Bucket=DEFAULT_BUCKET, Key=DEFAULT_DATA_KEY)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        print(f"[warn] s3 csv {DEFAULT_BUCKET}/{DEFAULT_DATA_KEY}: {e}")
        return None

DEFAULT_RAW = load_default_csv_from_s3()

# api
api = FastAPI(title="Stack Model API")

@api.get("/health")
def health():
    return {"status":"ok","model":type(MODEL).__name__,"features":FEATURES}

@api.post("/predict")
def predict(payload: Dict[str, list]):
    try:
        rows = payload.get("data", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("payload needs non-empty 'data' list")
        X = pd.DataFrame(rows)
        missing = [c for c in FEATURES if c not in X.columns]
        if missing: raise ValueError(f"missing features: {missing}")
        X = X[FEATURES]
        preds = MODEL.predict(X).tolist()
        out = {"predictions": preds}
        if hasattr(MODEL, "predict_proba"):
            out["predict_proba"] = MODEL.predict_proba(X).tolist()
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

VetiverAPI(VetiverModel(MODEL, "stack-model", prototype_data=X_PROTO), app=api, check_prototype=True)

# ui
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Upload tournaments CSV", accept=[".csv"], multiple=False),
        ui.input_action_button("go", "Run predictions"),
        ui.markdown(f"Default dataset: `s3://{DEFAULT_BUCKET}/{DEFAULT_DATA_KEY}`"),
        ui.input_checkbox("show_eval", "Compute metrics if labels available", True),
    ),
    ui.h3("Badminton Win Probability"),
    ui.card(ui.card_header("Predictions"), ui.output_table("pred_table")),
    ui.card(ui.card_header("Metrics"), ui.output_text("metrics")),
    ui.layout_columns(
        ui.card(ui.card_header("Win% (last 5) — Histogram"), ui.output_plot("gg_hist_win5")),
        ui.card(ui.card_header("Elo Diff — Density by Event"), ui.output_plot("gg_density_elodiff")),
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Elo Diff vs Win% (10)"), ui.output_plot("gg_scatter_elodiff_win10")),
        ui.card(ui.card_header("Calibration"), ui.output_plot("gg_calibration")),
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Top Players (gt)"), ui.output_ui("gt_top_players")),
        ui.card(ui.card_header("Recent H2H (gt)"), ui.output_ui("gt_recent_h2h")),
    ),
    title="Badminton H2H Predictor",
)

# serve
def shiny_server(input, output, session):

    @reactive.calc
    def raw_df():
        f = input.csv()
        if f:
            return pd.read_csv(f[0]["datapath"])
        return DEFAULT_RAW

    @reactive.calc
    def features_df():
        df_raw = raw_df()
        if df_raw is None:
            return None
        return build_features_from_raw(df_raw, precomp_pr=PRECOMP_PR)

    def api_predict_proba(X: pd.DataFrame) -> np.ndarray:
        try:
            payload = {"data": json.loads(X.to_json(orient="records"))}
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            obj = r.json()
            if "predict_proba" in obj:
                arr = np.asarray(obj["predict_proba"])
                return arr[:, 1] if arr.ndim == 2 else np.asarray(arr)
            return np.asarray(obj["predictions"])
        except Exception:
            if hasattr(MODEL, "predict_proba"):
                return MODEL.predict_proba(X)[:, 1]
            raw = np.asarray(MODEL.predict(X)).ravel()
            return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

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
        proba = api_predict_proba(X)
        out = feats[["player_id","opponent_id","date"]].copy()
        out["win_prob_player_id"] = np.asarray(proba).ravel()
        return out.sort_values("date")

    @output
    @render.text
    def metrics():
        if not input.show_eval(): return "(hidden)"
        feats = features_df()
        if feats is None or "win" not in feats.columns or feats["win"].isna().all():
            return "No labels; skipped."
        try:
            X = feats[FEATURES]
            p = api_predict_proba(X)
            y = feats["win"].astype(int)
            acc = accuracy_score(y, (np.asarray(p) >= 0.5).astype(int))
            return f"AUC={roc_auc_score(y,p):.3f} | Brier={brier_score_loss(y,p):.3f} | Acc={acc:.3f}"
        except Exception as e:
            return f"Unavailable: {e}"

    @output
    @render.plot
    def gg_hist_win5():
        feats = features_df()
        if feats is None or feats.empty or "win_pct_5" not in feats.columns: return
        df = feats.copy()
        df["win_pct_5"] = df["win_pct_5"].astype(float).clip(0, 1)
        p = ggplot(df, aes(x="win_pct_5")) + geom_histogram(bins=25) + labs(x="Win% (5)", y="Count", title="Recent win%") + theme_bw()
        return p.draw()

    @output
    @render.plot
    def gg_density_elodiff():
        feats = features_df()
        if feats is None or feats.empty or "elo_diff" not in feats.columns: return
        df = feats.copy()
        if "event" not in df.columns: df["event"] = "Unk"
        p = ggplot(df, aes(x="elo_diff")) + geom_density() + facet_wrap("~event", ncol=2) + labs(x="Elo Diff", y="Density", title="Elo diff by event") + theme_bw()
        return p.draw()

    @output
    @render.plot
    def gg_scatter_elodiff_win10():
        feats = features_df()
        if feats is None or feats.empty or not {"elo_diff","win_pct_10"}.issubset(set(feats.columns)): return
        df = feats.copy()
        df["win_pct_10"] = df["win_pct_10"].astype(float).clip(0, 1)
        p = ggplot(df, aes(x="elo_diff", y="win_pct_10")) + geom_point(alpha=0.25) + geom_smooth(method="loess", se=False) + labs(x="Elo Diff", y="Win% (10)", title="Elo vs recent win%") + theme_bw()
        return p.draw()

    @output
    @render.plot
    def gg_calibration():
        feats = features_df()
        if feats is None or feats.empty or "win" not in feats.columns: return
        X = feats[[c for c in FEATURES if c in feats.columns]].fillna(0)
        p = api_predict_proba(X)
        df = pd.DataFrame({"prob": p, "win": feats["win"].astype(int)})
        df["bin"] = pd.qcut(df["prob"], 10, duplicates="drop")
        cal = df.groupby("bin", observed=True).agg(mean_prob=("prob","mean"), actual_rate=("win","mean")).reset_index(drop=True)
        pc = ggplot(cal, aes(x="mean_prob", y="actual_rate")) + geom_point() + geom_smooth(method="loess", se=False) + labs(x="Pred prob", y="Observed", title="Calibration (10 bins)") + theme_bw()
        return pc.draw()

    @output
    @render.ui
    def gt_top_players():
        feats = features_df()
        if feats is None or feats.empty: return HTML("<i>No data</i>")
        agg = feats.groupby("player_id", as_index=False).agg(games=("win","size"), win_rate=("win","mean"))
        agg = agg[agg["games"] >= 5].sort_values(["win_rate","games"], ascending=[False,False]).head(15)
        agg.insert(1, "player", agg["player_id"].map(lambda pid: ID_TO_LABEL.get(str(pid), str(pid))))
        t = (GT(agg[["player","games","win_rate"]]).tab_header(title="Top Players (min 5)").fmt_percent(columns="win_rate", scale=1.0).style(loc.body, props=style(text_align="center")).style(loc.header, props=style(text_align="center", font_weight="bold")))
        return HTML(t.as_html())

    @output
    @render.ui
    def gt_recent_h2h():
        feats = features_df()
        if feats is None or feats.empty: return HTML("<i>No data</i>")
        df = feats.sort_values("date", ascending=False).head(20).copy()
        df["player"] = df["player_id"].map(lambda x: ID_TO_LABEL.get(str(x), str(x)))
        df["opponent"] = df["opponent_id"].map(lambda x: ID_TO_LABEL.get(str(x), str(x)))
        show = df[["date","player","opponent","win","elo_diff","pr_player","pr_opponent"]].copy()
        show["date"] = pd.to_datetime(show["date"]).dt.date
        t = (GT(show).tab_header(title="Recent Matches").style(loc.body, props=style(text_align="center")).style(loc.header, props=style(text_align="center", font_weight="bold")))
        return HTML(t.as_html())

app = App(app_ui, shiny_server)