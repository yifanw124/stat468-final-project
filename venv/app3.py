# app.py — Predictive Shiny app with default CSV + ggplots + full predictions table
from __future__ import annotations
import io, os, json, warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from shiny import App, reactive, render, ui
from plotnine import (
    ggplot, aes, geom_col, coord_flip, labs, theme_bw,
    scale_y_continuous
)

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
        ui.input_action_button("go", "Compute predictions"),
        ui.output_text("status"),
        ui.markdown(
            "If no file is uploaded, the app will try to use the CSV at:\n\n"
            "```\n"
            "/Users/yifanw124/STAT468/stat468-final-project/tournaments_2018_2025_June.csv\n"
            "```"
        ),
    ),
    ui.h3("Badminton Win Probability — Aggregates"),
    ui.layout_columns(
        ui.card(ui.card_header("Top Avg Win Probability (ggplot)"), ui.output_plot("gg_top_avg")),
        ui.card(ui.card_header("Lowest Avg Win Probability (ggplot)"), ui.output_plot("gg_bottom_avg")),
        ui.card(ui.card_header("Most Matches (ggplot)"), ui.output_plot("gg_most_matches")),
    ),
    ui.card(
        ui.card_header("Full Predictions Table"),
        ui.output_table("pred_table")
    ),
    title="Badminton H2H Predictor — Aggregates + Table",
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
        return try_load_default_csv()

    @reactive.calc
    def features_df() -> pd.DataFrame | None:
        df_raw = raw_df()
        if df_raw is None or df_raw.empty:
            return None
        use_pr = PRECOMP_PR if _BUNDLE and input.csv() is None else None
        feats = build_features_from_raw(df_raw, precomp_pr=use_pr)
        # Normalize IDs to strings
        feats["player_id"] = feats["player_id"].astype(str)
        feats["opponent_id"] = feats["opponent_id"].astype(str)
        return feats

    @reactive.calc
    def predictions_df() -> pd.DataFrame | None:
        _ = input.go()
        feats = features_df()
        if feats is None or feats.empty or not model_ready():
            return None
        X = feats[FEATURES].copy()
        proba = np.asarray(MODEL.predict_proba(X))[:, 1]
        out = feats[["player_id", "opponent_id", "date"]].copy()
        out["player"] = out["player_id"].map(lambda x: ID_TO_LABEL.get(x, x))
        out["opponent"] = out["opponent_id"].map(lambda x: ID_TO_LABEL.get(x, x))
        out["win_prob_player_id"] = proba
        out["date"] = pd.to_datetime(out["date"])
        return out

    @reactive.calc
    def agg_players() -> pd.DataFrame | None:
        df = predictions_df()
        if df is None or df.empty:
            return None
        agg = (
            df.groupby("player_id", as_index=False)
              .agg(avg_win_prob=("win_prob_player_id", "mean"), matches=("win_prob_player_id", "size"))
        )
        agg.insert(1, "player", agg["player_id"].map(lambda x: ID_TO_LABEL.get(x, x)))
        agg["avg_win_prob"] = agg["avg_win_prob"].astype(float)
        agg["matches"] = agg["matches"].astype(int)
        return agg

    @output
    @render.text
    def status():
        msgs = []
        msgs.append("Model: ✅ loaded" if model_ready() else "Model: ❌ not loaded (place stack_model.joblib beside app.py)")
        df = raw_df()
        msgs.append("Data: ✅ loaded" if (df is not None and not df.empty) else "Data: ❌ not found (upload CSV or ensure default path exists)")
        return " | ".join(msgs)

    # ----- ggplot Bar Charts -----
    @output
    @render.plot
    def gg_top_avg():
        agg = agg_players()
        if agg is None or agg.empty:
            return
        df = agg.sort_values(["avg_win_prob", "matches"], ascending=[False, False]).head(15).copy()
        df["player"] = pd.Categorical(df["player"], categories=df.sort_values("avg_win_prob")["player"], ordered=True)
        p = (
            ggplot(df, aes(x="player", y="avg_win_prob"))
            + geom_col()
            + coord_flip()
            + scale_y_continuous(limits=[0, 1])
            + labs(x="", y="Avg win prob", title="Top Avg Win Probability")
            + theme_bw()
        )
        return p.draw()

    @output
    @render.plot
    def gg_bottom_avg():
        agg = agg_players()
        if agg is None or agg.empty:
            return
        df = agg.sort_values(["avg_win_prob", "matches"], ascending=[True, False]).head(15).copy()
        df["player"] = pd.Categorical(df["player"], categories=df.sort_values("avg_win_prob")["player"], ordered=True)
        p = (
            ggplot(df, aes(x="player", y="avg_win_prob"))
            + geom_col()
            + coord_flip()
            + scale_y_continuous(limits=[0, 1])
            + labs(x="", y="Avg win prob", title="Lowest Avg Win Probability")
            + theme_bw()
        )
        return p.draw()

    @output
    @render.plot
    def gg_most_matches():
        agg = agg_players()
        if agg is None or agg.empty:
            return
        df = agg.sort_values(["matches", "avg_win_prob"], ascending=[False, False]).head(15).copy()
        df["player"] = pd.Categorical(df["player"], categories=df.sort_values("matches")["player"], ordered=True)
        p = (
            ggplot(df, aes(x="player", y="matches"))
            + geom_col()
            + coord_flip()
            + labs(x="", y="# matches", title="Most Matches")
            + theme_bw()
        )
        return p.draw()

    # ----- Full predictions table (original per-row info) -----
    @output
    @render.table
    def pred_table():
        df = predictions_df()
        if df is None or df.empty:
            return pd.DataFrame()
        show = df.copy()
        show["date"] = show["date"].dt.date
        show = show.sort_values("date")
        cols = ["date", "player", "opponent", "win_prob_player_id", "player_id", "opponent_id"]
        return show[cols]

app = App(app_ui, server)
