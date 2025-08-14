# app.py — Predictive Shiny app with local default CSV + H2H single-number prediction
from __future__ import annotations
import io, os, json, warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from shiny import App, reactive, render, ui

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- config ----------
HERE = Path(__file__).resolve().parent

# Model file (ship with app OR adjust to your path)
DEFAULT_LOCAL_MODEL = HERE / "stack_model.joblib"

# Default CSV fallback (your local absolute path)
DEFAULT_LOCAL_CSV = Path("/Users/yifanw124/STAT468/stat468-final-project/tournaments_2018_2025_June.csv")

FEATURE_SPEC_FILE = HERE / "feature_spec.json"

def try_load_model() -> Dict[str, Any] | None:
    if DEFAULT_LOCAL_MODEL.exists():
        try:
            with open(DEFAULT_LOCAL_MODEL, "rb") as f:
                return joblib.load(f)
        except Exception as e:
            print(f"[warn] Local model load failed: {e}")
    return None

def try_load_default_csv() -> pd.DataFrame | None:
    if DEFAULT_LOCAL_CSV.exists():
        try:
            return pd.read_csv(DEFAULT_LOCAL_CSV)
        except Exception as e:
            print(f"[warn] Default CSV load failed: {e}")
    return None

def load_feature_spec() -> dict | None:
    if FEATURE_SPEC_FILE.exists():
        try:
            with open(FEATURE_SPEC_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

# Load once
_BUNDLE = try_load_model()
_FEATURE_SPEC = load_feature_spec()

RAW_MODEL = _BUNDLE["model"] if isinstance(_BUNDLE, dict) and "model" in _BUNDLE else None
if _FEATURE_SPEC and "features" in _FEATURE_SPEC:
    FEATURES: List[str] = list(_FEATURE_SPEC["features"])
elif isinstance(_BUNDLE, dict) and "features" in _BUNDLE and _BUNDLE["features"]:
    FEATURES = list(_BUNDLE["features"])
else:
    FEATURES = ["elo_diff","win_pct_5","win_pct_10","win_pct_20","h2h_decay","h2h_adj","pr_player","pr_opponent"]

ID_TO_LABEL: Dict[str, str] = dict(_BUNDLE.get("id_to_label", {})) if isinstance(_BUNDLE, dict) else {}
PRECOMP_PR = _BUNDLE.get("pagerank", {}) if isinstance(_BUNDLE, dict) else {}

class WrappedEstimator:
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

MODEL = WrappedEstimator(RAW_MODEL) if RAW_MODEL is not None else None

# ---------- feature builder (same as training) ----------
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

    from collections import defaultdict as dd
    elo = dd(lambda: DEFAULT_ELO)
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

    # Ensure feature columns exist
    for c in FEATURES:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    df_feat[FEATURES] = df_feat[FEATURES].fillna(0)

    keep = ["player_id","opponent_id","date","event"] + FEATURES + ["win"]
    return df_feat[keep]

# ---------- UI ----------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Upload tournaments CSV (same schema)", accept=[".csv"], multiple=False),
        ui.input_action_button("go", "Run predictions"),
        ui.input_selectize("player", "Player", choices=[], multiple=False),
        ui.input_selectize("opponent", "Opponent", choices=[], multiple=False),
        ui.output_text("status"),
        ui.output_text("h2h_pred"),
        ui.markdown(
            "If no file is uploaded, the app will try to use the CSV at:\n\n"
            "```\n"
            "/Users/yifanw124/STAT468/stat468-final-project/tournaments_2018_2025_June.csv\n"
            "```"
        ),
    ),
    ui.h3("Badminton Win Probability — Predictions"),
    ui.card(ui.card_header("Predictions"), ui.output_table("pred_table")),
    title="Badminton H2H Predictor",
)

# ---------- server ----------
def server(input, output, session):
    @reactive.calc
    def model_ready() -> bool:
        return MODEL is not None

    @reactive.calc
    def raw_df() -> pd.DataFrame | None:
        f = input.csv()
        if f:
            return pd.read_csv(f[0]["datapath"])
        # fallback to your absolute local file
        return try_load_default_csv()

    @reactive.calc
    def features_df() -> pd.DataFrame | None:
        df_raw = raw_df()
        if df_raw is None or df_raw.empty:
            return None
        use_pr = PRECOMP_PR if _BUNDLE and input.csv() is None else None
        feats = build_features_from_raw(df_raw, precomp_pr=use_pr)

        # --- Normalize IDs to strings globally to avoid filter mismatches ---
        feats["player_id"] = feats["player_id"].astype(str)
        feats["opponent_id"] = feats["opponent_id"].astype(str)
        return feats

    @output
    @render.text
    def status():
        if not model_ready():
            return "⚠️ Model not loaded. Place stack_model.joblib beside app.py."
        if raw_df() is None:
            return "⚠️ No data available. Upload a CSV or ensure the default path exists."
        return "Model and data loaded ✔️"

    # Populate dropdowns
    @reactive.effect
    def _populate_dropdowns():
        feats = features_df()
        if feats is None or feats.empty:
            ui.update_selectize("player", choices=[], selected=None)
            ui.update_selectize("opponent", choices=[], selected=None)
            return
        players = sorted(feats["player_id"].unique().tolist())
        labeled = [(ID_TO_LABEL.get(p, p), p) for p in players]  # (label, value=str id)
        ui.update_selectize("player", choices=labeled, server=True)
        ui.update_selectize("opponent", choices=labeled, server=True)

    def _filter_df(df_pred: pd.DataFrame) -> pd.DataFrame:
        p_sel = input.player() or ""
        o_sel = input.opponent() or ""
        if p_sel:
            df_pred = df_pred[df_pred["player_id"] == p_sel]
        if o_sel:
            df_pred = df_pred[df_pred["opponent_id"] == o_sel]
        return df_pred

    # Full prediction table
    @output
    @render.table
    def pred_table():
        _ = input.go()
        feats = features_df()
        if feats is None or feats.empty:
            return pd.DataFrame()
        if not model_ready():
            return pd.DataFrame({"Error": ["Model not loaded. See status above."]})

        X = feats[FEATURES].copy()
        proba = np.asarray(MODEL.predict_proba(X))[:, 1]

        out = feats[["player_id","opponent_id","date"]].copy()
        # IDs already normalized to strings
        out["player"]   = out["player_id"].map(lambda x: ID_TO_LABEL.get(x, x))
        out["opponent"] = out["opponent_id"].map(lambda x: ID_TO_LABEL.get(x, x))
        out["win_prob_player_id"] = proba
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.sort_values("date")

        out = _filter_df(out)
        return out[["date","player","opponent","win_prob_player_id","player_id","opponent_id"]]

    # Head-to-head prediction: single number (Player → Opponent), most recent row
    @output
    @render.text
    def h2h_pred():
        p_sel, o_sel = input.player(), input.opponent()
        # Only show a value when both are selected
        if not p_sel or not o_sel:
            return ""
        feats = features_df()
        if MODEL is None or feats is None or feats.empty:
            return "NA"

        pair = (
            feats[(feats["player_id"] == p_sel) & (feats["opponent_id"] == o_sel)]
            .sort_values("date")
        )
        if pair.empty:
            return "NA"

        x = pair.iloc[[-1]][FEATURES]
        try:
            prob = float(MODEL.predict_proba(x)[:, 1][0])
            return f"{prob:.3f}"  # single number
        except Exception:
            return "NA"

app = App(app_ui, server)
