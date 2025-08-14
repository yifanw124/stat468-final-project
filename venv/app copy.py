# app.py
from __future__ import annotations
import io, os, pickle, warnings, json
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import logging
import joblib

import numpy as np
import pandas as pd
import networkx as nx
from fastapi import FastAPI, HTTPException
from shiny import App, reactive, render, ui
from plotnine import ggplot, aes, geom_histogram, geom_density, geom_point, geom_smooth, facet_wrap, labs, theme_bw
from great_tables import GT, style, loc
from htmltools import HTML
import requests
import duckdb

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

warnings.filterwarnings("ignore", category=UserWarning)

# --------- config & paths ----------
HERE = Path(__file__).resolve().parent
DEFAULT_LOCAL_MODEL = Path("/Users/yifanw124/STAT468/stat468-final-project/stack_model.joblib")
DEFAULT_LOCAL_CSV = Path("/Users/yifanw124/STAT468/stat468-final-project/tournaments_2018_2025_June.csv")
FEATURE_SPEC_FILE = HERE / "feature_spec.json" # New path for the feature spec

DEFAULT_BUCKET = os.getenv("MODEL_BUCKET", "")
DEFAULT_MODEL_KEY = os.getenv("MODEL_KEY", "")
DEFAULT_DATA_KEY = os.getenv("DATA_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

try:
    import boto3
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False

def s3_client():
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed")
    return boto3.client("s3", region_name=AWS_REGION)

# --------- load model bundle ----------
def load_model_bundle() -> dict:
    if HAS_BOTO3 and DEFAULT_BUCKET and DEFAULT_MODEL_KEY:
        try:
            obj = s3_client().get_object(Bucket=DEFAULT_BUCKET, Key=DEFAULT_MODEL_KEY)
            return joblib.load(io.BytesIO(obj["Body"].read()))
        except Exception as e:
            print(f"[warn] S3 model load failed: {e}")

    if DEFAULT_LOCAL_MODEL.exists():
        with open(DEFAULT_LOCAL_MODEL, "rb") as f:
            return joblib.load(f)

    raise FileNotFoundError("No model bundle found in S3 or local path")

# --------- load feature spec ----------
def load_feature_spec() -> dict:
    if FEATURE_SPEC_FILE.exists():
        with open(FEATURE_SPEC_FILE, "r") as f:
            return json.load(f)
    return None

BUNDLE = load_model_bundle()
RAW_MODEL = BUNDLE["model"]

FEATURE_SPEC = load_feature_spec()
if FEATURE_SPEC:
    FEATURES: List[str] = FEATURE_SPEC.get("features", [])
else:
    FEATURES: List[str] = list(BUNDLE.get("features", []))
    if not FEATURES:
        FEATURES = ["elo_diff","win_pct_5","win_pct_10","win_pct_20","h2h_decay","h2h_adj","pr_player","pr_opponent"]

ID_TO_LABEL: Dict[str, str] = dict(BUNDLE.get("id_to_label", {}))
PRECOMP_PR = BUNDLE.get("pagerank", {})

class WrappedEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, inner):
        self.inner = inner
    def predict(self, X):
        return self.inner.predict(X)
    def predict_proba(self, X):
        if hasattr(self.inner, "predict_proba"):
            return self.inner.predict_proba(X)
        if hasattr(self.inner, "decision_function"):
            z = np.asarray(self.inner.decision_function(X))
            p = 1 / (1 + np.exp(-z))
        else:
            y = np.asarray(self.inner.predict(X)).ravel()
            p = (y - y.min()) / (y.max() - y.min() + 1e-9)
        return np.vstack([1 - p, p]).T

MODEL = WrappedEstimator(RAW_MODEL)

X_PROTO = pd.DataFrame([{c: 0.0 for c in FEATURES}]).astype({c: "float64" for c in FEATURES})

# --------- default raw CSV loader ----------
def load_default_raw() -> pd.DataFrame | None:
    if HAS_BOTO3 and DEFAULT_BUCKET and DEFAULT_DATA_KEY:
        try:
            obj = s3_client().get_object(Bucket=DEFAULT_BUCKET, Key=DEFAULT_DATA_KEY)
            return pd.read_csv(io.BytesIO(obj["Body"].read()))
        except Exception as e:
            print(f"[warn] S3 CSV load failed: {e}")

    if DEFAULT_LOCAL_CSV.exists():
        return pd.read_csv(DEFAULT_LOCAL_CSV)

    print("[warn] No default CSV available")
    return None

DEFAULT_RAW = load_default_raw()

# --------- feature builder (mirrors training) ----------
DEFAULT_ELO = 1200
K = 32
REQUIRED_COLS = ["date","event","player1","player2","winner","team1_total_points","team2_total_points","tournament_name"]

def _expected_score(rA, rB):
    return 1 / (1 + 10 ** ((rB - rA) / 400))
def _update_elo(rA, rB, outA):
    eA = _expected_score(rA, rB)
    return rA + K*(outA - eA), rB + K*((1-outA) - (1-eA))

def build_features_from_raw(df_raw: pd.DataFrame, precomp_pr: Dict[str,float] | None=None) -> pd.DataFrame:
    miss = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df = df_raw[df_raw["event"].str.contains("MS|WS", regex=True)].copy()
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

    alpha = 0.1
    df_feat = df_feat.sort_values("date")
    df_feat["h2h_decay"] = (
        df_feat.groupby(["player_id","opponent_id"], sort=False)["win"]
               .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
               .reset_index(level=[0,1], drop=True)
    )

    df_feat["h2h_adj"] = df_feat["h2h_decay"] * (df_feat["elo_opponent"] / (df_feat["elo_player"] + 1e-6))

    if precomp_pr is None:
        G = nx.DiGraph()
        for _, r in df_feat.iterrows():
            if r["win"] == 1:
                G.add_edge(r["opponent_id"], r["player_id"])
        pr = nx.pagerank(G, alpha=0.85)
    else:
        pr = {str(k): float(v) for k, v in (precomp_pr or {}).items()}

    df_feat["pr_player"]   = df_feat["player_id"].map(lambda x: pr.get(str(x), 0.0)).fillna(0.0)
    df_feat["pr_opponent"] = df_feat["opponent_id"].map(lambda x: pr.get(str(x), 0.0)).fillna(0.0)

    for c in FEATURES:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    df_feat[FEATURES] = df_feat[FEATURES].fillna(0)

    keep = ["player_id","opponent_id","date","event"] + FEATURES + ["win"]
    return df_feat[keep]

# --------- API ----------
api = FastAPI(title="Stack Model API")

@api.get("/health")
def health():
    return {"status":"ok","model":type(MODEL.inner).__name__,"features":FEATURES}

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
        out = {"predictions": MODEL.predict(X).tolist()}
        if hasattr(MODEL, "predict_proba"):
            out["predict_proba"] = MODEL.predict_proba(X).tolist()
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------- UI ----------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Upload tournaments CSV (same schema)", accept=[".csv"], multiple=False),
        ui.input_action_button("go", "Run predictions"),
        ui.markdown("If no file uploaded, app uses the **default** training dataset."),
        ui.input_checkbox("show_eval", "Show metrics (if labels available)", True),
        ui.input_selectize("player", "Player", choices=[], multiple=False, options={"create": False}),
        ui.input_selectize("opponent", "Opponent (optional)", choices=[], multiple=False, options={"create": False}),
        ui.output_text("api_log_report"),
    ),
    ui.h3("Badminton Win Probability"),
    ui.card(ui.card_header("Predictions (filtered)"), ui.output_table("pred_table")),
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

# --------- server ----------
def shiny_server(input, output, session):

    log_messages = reactive.Value("Ready.")
    
    @output
    @render.text
    def api_log_report():
        return log_messages.get()

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
        use_pr = PRECOMP_PR if input.csv() is None else None
        return build_features_from_raw(df_raw, precomp_pr=use_pr)

    @reactive.event(input.go)
    def _run():
        pass

    def _filter_by_selection(df_pred: pd.DataFrame) -> pd.DataFrame:
        p_sel = input.player()
        o_sel = input.opponent()
        if not p_sel:
            return df_pred
        
        con = duckdb.connect(':memory:')
        con.register('data_to_filter', df_pred)
        
        query = f"SELECT * FROM data_to_filter WHERE player_id = '{p_sel}'"
        if o_sel:
            query += f" AND opponent_id = '{o_sel}'"

        return con.execute(query).fetchdf()

    @output
    @render.table
    def pred_table():
        _ = input.go()
        
        feats = features_df()
        if feats is None or feats.empty:
            return pd.DataFrame()

        try:
            X = feats[FEATURES]
            log_messages.set(f"Sending API request to {API_URL}...")

            response = requests.post(API_URL, json={"data": X.to_dict(orient="records")})
            response.raise_for_status()

            predictions = response.json()
            proba = np.asarray(predictions["predict_proba"])[:, 1]
            log_messages.set("API call successful. Predictions received.")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API call failed: {e}. Check if the FastAPI service is running."
            log_messages.set(error_msg)
            warnings.warn(error_msg, UserWarning)
            return pd.DataFrame({"Error": [error_msg]})

        out = feats[["player_id","opponent_id","date"]].copy()
        out["player"] = out["player_id"].astype(str).map(lambda x: ID_TO_LABEL.get(x, x))
        out["opponent"] = out["opponent_id"].astype(str).map(lambda x: ID_TO_LABEL.get(x, x))
        out["win_prob_player_id"] = np.asarray(proba).ravel()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.sort_values("date")
        out = _filter_by_selection(out)

        cols = ["date","player","opponent","win_prob_player_id","player_id","opponent_id"]
        return out[cols]

    @reactive.effect
    def _populate_dropdowns():
        feats = features_df()
        if feats is None or feats.empty:
            ui.update_selectize("player", choices=[], selected=None)
            ui.update_selectize("opponent", choices=[], selected=None)
            return
        players = sorted(set(feats["player_id"].astype(str).tolist()))
        labeled = [(ID_TO_LABEL.get(p, p), p) for p in players]
        ui.update_selectize("player", choices=labeled, server=True)
        ui.update_selectize("opponent", choices=[("", "")] + labeled, server=True)

    @output
    @render.text
    def metrics():
        if not input.show_eval():
            return "(hidden)"
        feats = features_df()
        if feats is None or "win" not in feats.columns or feats["win"].isna().all():
            return "No labels; skipped."
        try:
            X = feats[FEATURES]
            p = MODEL.predict_proba(X)[:, 1]
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
        p = MODEL.predict_proba(X)[:, 1]
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
        t = (GT(agg[["player","games","win_rate"]])
             .tab_header(title="Top Players (min 5)")
             .fmt_percent(columns="win_rate", scale=1.0)
             .style(loc.body, props=style(text_align="center"))
             .style(loc.header, props=style(text_align="center", font_weight="bold")))
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
        t = (GT(show)
             .tab_header(title="Recent Matches")
             .style(loc.body, props=style(text_align="center"))
             .style(loc.header, props=style(text_align="center", font_weight="bold")))
        return HTML(t.as_html())

app = App(app_ui, shiny_server)