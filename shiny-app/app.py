from __future__ import annotations

import os, time, logging
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")

import io, json, warnings
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from shiny import App, reactive, render, ui
from plotnine import ggplot, aes, geom_col, coord_flip, labs, theme_bw, scale_y_continuous
import requests
from dotenv import load_dotenv

import duckdb, os

con = duckdb.connect(":memory:")
con.execute("INSTALL httpfs; LOAD httpfs;")

# Pass AWS credentials into DuckDB's S3 layer
con.execute("SET s3_region = ?", [os.getenv("AWS_DEFAULT_REGION")])
con.execute("SET s3_access_key_id = ?", [os.getenv("AWS_ACCESS_KEY_ID")])
con.execute("SET s3_secret_access_key = ?", [os.getenv("AWS_SECRET_ACCESS_KEY")])

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("app")

def _redact(s: str | None) -> str:
    if not s: return ""
    try:
        from urllib.parse import urlparse
        u = urlparse(s)
        host = u.hostname or ""
        return f"{u.scheme}://{host}"
    except Exception:
        return "<redacted>"

# ---------- config ----------
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env") 

DEFAULT_LOCAL_MODEL = HERE / "vetiver_model.joblib"
DEFAULT_LOCAL_CSV   = HERE / "tournaments_2018_2025_June.csv"
FEATURE_SPEC_FILE   = HERE / "feature_spec.json"

S3_BUCKET        = os.getenv("S3_BUCKET", "").strip()
S3_MODEL_KEY     = os.getenv("S3_MODEL_KEY", "").strip()
S3_DATA_KEY      = os.getenv("S3_DATA_KEY", "").strip()
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX", "outputs")
AWS_REGION       = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
S3_ENDPOINT      = os.getenv("AWS_S3_ENDPOINT", "").strip()
S3_ENABLED       = bool(S3_BUCKET and (S3_MODEL_KEY or S3_DATA_KEY))

PERSIST_PREDICTIONS_S3 = os.getenv("PERSIST_PREDICTIONS_S3", "false").lower() == "true"
PERSIST_FORMAT         = os.getenv("PERSIST_FORMAT", "parquet").lower()

EC2_PUBLIC_IPV4 = os.getenv("EC2_PUBLIC_IPV4", "").strip()
API_BASE_URL    = os.getenv("REMOTE_API_BASE_URL", f"http://{EC2_PUBLIC_IPV4}:8000" if EC2_PUBLIC_IPV4 else "")
if not API_BASE_URL:
    API_BASE_URL = "http://127.0.0.1:8000"
API_PRED_ENDPOINT = os.getenv("REMOTE_API_PRED_ENDPOINT", "/predict_proba")
USE_REMOTE        = os.getenv("USE_REMOTE", "true").lower() == "true"
BATCH_SIZE        = int(os.getenv("REMOTE_BATCH_SIZE", "500"))
REQUEST_TIMEOUT   = float(os.getenv("REMOTE_TIMEOUT", "15"))
TRY_VETIVER_PREDICT_FALLBACK = os.getenv("TRY_VETIVER_PREDICT_FALLBACK", "false").lower() == "true"

log.info("boot S3_ENABLED=%s bucket=%s model_key=%s data_key=%s region=%s",
         S3_ENABLED, S3_BUCKET or "<unset>", S3_MODEL_KEY or "<unset>",
         S3_DATA_KEY or "<unset>", AWS_REGION)
log.info("boot REMOTE_API=%s endpoint=%s use_remote=%s",
         _redact(API_BASE_URL), API_PRED_ENDPOINT, USE_REMOTE)
log.debug("boot ENV has AWS keys? %s %s",
          bool(os.getenv("AWS_ACCESS_KEY_ID")), bool(os.getenv("AWS_SECRET_ACCESS_KEY")))

# ---------- DuckDB helpers (S3 via httpfs) ----------
def duckdb_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region = ?", [AWS_REGION])
    if os.getenv("AWS_ACCESS_KEY_ID"):
        con.execute("SET s3_access_key_id = ?", [os.getenv("AWS_ACCESS_KEY_ID")])
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        con.execute("SET s3_secret_access_key = ?", [os.getenv("AWS_SECRET_ACCESS_KEY")])
    if os.getenv("AWS_SESSION_TOKEN"):
        con.execute("SET s3_session_token = ?", [os.getenv("AWS_SESSION_TOKEN")])
    if S3_ENDPOINT:
        con.execute("SET s3_endpoint = ?", [S3_ENDPOINT])
        con.execute("SET s3_url_style = 'v2';")
    con.execute("SET s3_use_ssl = true;")
    return con

def load_csv_from_s3_duckdb(bucket: str, key: str) -> pd.DataFrame | None:
    if not (bucket and key):
        return None
    uri = f"s3://{bucket}/{key}"
    try:
        t0 = time.time()
        con = duckdb_conn()
        df = con.execute("SELECT * FROM read_csv_auto(?, HEADER=TRUE)", [uri]).df()
        log.info("s3.csv.read ok uri=%s rows=%s ms=%d", uri, len(df), int((time.time()-t0)*1000))
        return df
    except Exception as e:
        log.warning("s3.csv.read failed uri=%s err=%s", uri, e)
        return None

def save_df_to_s3_duckdb(df: pd.DataFrame, bucket: str, key: str, fmt: str = "parquet") -> bool:
    if df is None or df.empty:
        return False
    uri = f"s3://{bucket}/{key}"
    try:
        t0 = time.time()
        con = duckdb_conn()
        con.register("df", df)
        if fmt == "csv":
            con.execute(f"COPY df TO '{uri}' (FORMAT CSV, HEADER, DELIMITER ',')")
        else:
            con.execute(f"COPY df TO '{uri}' (FORMAT PARQUET)")
        log.info("s3.df.write ok uri=%s rows=%s ms=%d", uri, len(df), int((time.time()-t0)*1000))
        return True
    except Exception as e:
        log.warning("s3.df.write failed uri=%s err=%s", uri, e)
        return False


# ---------- Model & data loaders (S3 first, then local) ----------
def try_load_model() -> Dict[str, Any] | Any | None:
    obj = None
    if S3_ENABLED and S3_MODEL_KEY:
        try:
            import boto3, tempfile
            cli = boto3.client("s3", region_name=AWS_REGION)
            tmp_path = Path(tempfile.gettempdir()) / ("model_" + Path(S3_MODEL_KEY).name)
            cli.download_file(S3_BUCKET, S3_MODEL_KEY, str(tmp_path))
            with open(tmp_path, "rb") as f:
                obj = joblib.load(f)
            log.info("s3.model.load ok s3://%s/%s", S3_BUCKET, S3_MODEL_KEY)
        except Exception as e:
            log.warning("s3.model.load failed s3://%s/%s err=%s", S3_BUCKET, S3_MODEL_KEY, e)

    if obj is None and DEFAULT_LOCAL_MODEL.exists():
        try:
            with open(DEFAULT_LOCAL_MODEL, "rb") as f:
                obj = joblib.load(f)
            log.info("local.model.load ok path=%s", DEFAULT_LOCAL_MODEL)
        except Exception as e:
            log.warning("local.model.load failed path=%s err=%s", DEFAULT_LOCAL_MODEL, e)

    if obj is None:
        log.warning("model.unavailable")
    return obj

def try_load_default_csv() -> pd.DataFrame | None:
    df = None
    if S3_ENABLED and S3_DATA_KEY:
        df = load_csv_from_s3_duckdb(S3_BUCKET, S3_DATA_KEY)
    if df is None and DEFAULT_LOCAL_CSV.exists():
        try:
            df = pd.read_csv(DEFAULT_LOCAL_CSV)
            log.info("local.csv.read ok path=%s rows=%s", DEFAULT_LOCAL_CSV, len(df))
        except Exception as e:
            log.warning("local.csv.read failed path=%s err=%s", DEFAULT_LOCAL_CSV, e)
    if df is None:
        log.warning("data.unavailable (no S3 nor local CSV)")
    return df

def load_feature_spec() -> dict | None:
    if FEATURE_SPEC_FILE.exists():
        try:
            with open(FEATURE_SPEC_FILE, "r") as f:
                js = json.load(f)
            log.info("feature_spec.load ok keys=%s", list(js.keys()))
            return js
        except Exception as e:
            log.warning("feature_spec.load failed err=%s", e)
    return None

# ---------- bootstrap & features ----------
_BUNDLE = try_load_model()
_FEATURE_SPEC = load_feature_spec()

RAW_MODEL = None
ID_TO_LABEL: Dict[str, str] = {}
PRECOMP_PR: Dict[str, float] = {}

if isinstance(_BUNDLE, dict):
    ID_TO_LABEL = dict(_BUNDLE.get("id_to_label", {}))
    PRECOMP_PR  = _BUNDLE.get("pagerank", {})
    inner = _BUNDLE.get("model")
else:
    inner = _BUNDLE

if inner is not None and not any(hasattr(inner, a) for a in ("predict", "predict_proba", "decision_function")):
    if hasattr(inner, "model"):
        inner = inner.model
    elif hasattr(inner, "object"):
        inner = inner.object

RAW_MODEL = inner

if _FEATURE_SPEC and "features" in _FEATURE_SPEC:
    FEATURES: List[str] = list(_FEATURE_SPEC["features"])
elif isinstance(_BUNDLE, dict) and _BUNDLE.get("features"):
    FEATURES = list(_BUNDLE["features"])
else:
    FEATURES = ["elo_diff","win_pct_5","win_pct_10","win_pct_20","h2h_decay","h2h_adj","pr_player","pr_opponent"]

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
log.info("boot model_loaded=%s features=%s", bool(MODEL), FEATURES)

# ---------- feature builder (unchanged logic) ----------
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
    for c in FEATURES:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    df_feat[FEATURES] = (
        df_feat[FEATURES]
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
          .clip(lower=-1e9, upper=1e9)
          .astype(float)
    )
    keep = ["player_id","opponent_id","date","event"] + FEATURES + ["win"]
    return df_feat[keep]

# ---------- Remote call helpers (with logging) ----------
def _post_predict(records: list[dict]) -> list[float]:
    url = f"{API_BASE_URL.rstrip('/')}{API_PRED_ENDPOINT}"
    t0 = time.time()
    try:
        r = requests.post(url, json=records, timeout=REQUEST_TIMEOUT)
        status = r.status_code
        if status == 404 and TRY_VETIVER_PREDICT_FALLBACK:
            vurl = f"{API_BASE_URL.rstrip('/')}/predict"
            log.info("api.fallback url=%s", _redact(vurl))
            r = requests.post(vurl, json={"dataframe_records": records}, timeout=REQUEST_TIMEOUT)
            status = r.status_code
        r.raise_for_status()
        js = r.json()
        lat_ms = int((time.time() - t0) * 1000)
        log.info("api.post ok url=%s status=%s n=%s ms=%d", _redact(url), status, len(records), lat_ms)
        if isinstance(js, dict) and isinstance(js.get("probabilities"), list):
            return js["probabilities"]
        raise RuntimeError(f"Unexpected response JSON keys: {list(js)[:5]}")
    except Exception as e:
        lat_ms = int((time.time() - t0) * 1000)
        log.error("api.post fail url=%s n=%s ms=%d err=%s", _redact(url), len(records), lat_ms, e)
        raise

def _clean_features_for_api(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    Xc = X[features].copy()
    for c in features:
        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    Xc = (Xc.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e9, 1e9).astype(float))
    return Xc

def remote_predict_proba(X: pd.DataFrame, features: list[str]) -> np.ndarray:
    Xp = _clean_features_for_api(X, features).round(6)
    rows = Xp.to_dict(orient="records")
    out: list[float] = []
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i:i+BATCH_SIZE]
        probs = _post_predict(chunk)
        out.extend(probs)
    return np.asarray(out, dtype=float)

# ---------- UI ----------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Upload tournaments CSV (same schema)", accept=[".csv"], multiple=False),
        ui.input_action_button("go", "Compute predictions"),
        ui.output_text("status"),
        ui.markdown("If no file is uploaded, the app will try to use S3 (or local fallback)."),
    ),
    ui.h3("Badminton Win Probability — Aggregates"),
    ui.layout_columns(
        ui.card(ui.card_header("Top Avg Win Probability (ggplot)"), ui.output_plot("gg_top_avg")),
        ui.card(ui.card_header("Lowest Avg Win Probability (ggplot)"), ui.output_plot("gg_bottom_avg")),
        ui.card(ui.card_header("Most Matches (ggplot)"), ui.output_plot("gg_most_matches")),
    ),
    ui.card(ui.card_header("Full Predictions Table"), ui.output_table("pred_table")),
    title="Badminton H2H Predictor — Aggregates + Table",
)

# ---------- server ----------
def server(input, output, session):
    @reactive.effect
    def _log_clicks():
        _ = input.go()
        log.info("ui.click button=go")

    @reactive.calc
    def model_ready() -> bool:
        return MODEL is not None

    @reactive.calc
    def raw_df() -> pd.DataFrame | None:
        f = input.csv()
        if f:
            try:
                fp = f[0]["datapath"]
                df = pd.read_csv(fp)
                log.info("upload.csv.read name=%s size=%s rows=%s", f[0].get("name"), f[0].get("size"), len(df))
                return df
            except Exception as e:
                log.error("upload.csv.read.fail name=%s err=%s", f[0].get("name"), e)
                return None
        df = try_load_default_csv()
        return df

    @reactive.calc
    def features_df() -> pd.DataFrame | None:
        df_raw = raw_df()
        if df_raw is None or df_raw.empty:
            return None
        use_pr = PRECOMP_PR if _BUNDLE and input.csv() is None else None
        feats = build_features_from_raw(df_raw, precomp_pr=use_pr)
        feats["player_id"] = feats["player_id"].astype(str)
        feats["opponent_id"] = feats["opponent_id"].astype(str)
        log.debug("features.built rows=%s cols=%s", len(feats), list(feats.columns))
        return feats

    @reactive.calc
    def predictions_df() -> pd.DataFrame | None:
        _ = input.go()
        feats = features_df()
        if feats is None or feats.empty:
            return None
        X = feats[FEATURES].copy()
        proba = None
        if USE_REMOTE:
            try:
                proba = remote_predict_proba(X, FEATURES)
            except Exception as e:
                if MODEL is not None:
                    log.warning("predict.remote.fail fallback_local err=%s", e)
                else:
                    ui.notification_show(f"Remote prediction failed: {e}", duration=8, type="error")
                    return None
        if proba is None:
            if MODEL is None:
                ui.notification_show("No model available. Enable remote API or provide S3/local model.", duration=8, type="warning")
                log.error("predict.local.unavailable")
                return None
            proba = np.asarray(MODEL.predict_proba(X))[:, 1]
            log.info("predict.local.ok n=%s", len(proba))
        out = feats[["player_id", "opponent_id", "date"]].copy()
        out["player"] = out["player_id"].map(lambda x: ID_TO_LABEL.get(x, x))
        out["opponent"] = out["opponent_id"].map(lambda x: ID_TO_LABEL.get(x, x))
        out["win_prob_player_id"] = proba
        out["date"] = pd.to_datetime(out["date"])
        if PERSIST_PREDICTIONS_S3 and S3_BUCKET:
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            ext = "parquet" if PERSIST_FORMAT == "parquet" else "csv"
            key = f"{S3_OUTPUT_PREFIX.rstrip('/')}/predictions_{ts}.{ext}"
            ok = save_df_to_s3_duckdb(out, S3_BUCKET, key, fmt=PERSIST_FORMAT)
            if ok:
                log.info("predictions.persisted s3://%s/%s", S3_BUCKET, key)
            else:
                log.warning("predictions.persist.fail s3://%s/%s", S3_BUCKET, key)
        return out

    @reactive.calc
    def agg_players() -> pd.DataFrame | None:
        df = predictions_df()
        if df is None or df.empty:
            return None
        agg = (df.groupby("player_id", as_index=False)
                 .agg(avg_win_prob=("win_prob_player_id","mean"),
                      matches=("win_prob_player_id","size")))
        agg.insert(1, "player", agg["player_id"].map(lambda x: ID_TO_LABEL.get(x, x)))
        agg["avg_win_prob"] = agg["avg_win_prob"].astype(float)
        agg["matches"] = agg["matches"].astype(int)
        return agg

    @output
    @render.text
    def status():
        msgs = []
        msgs.append(f"Remote API: {_redact(API_BASE_URL)}{API_PRED_ENDPOINT}" if USE_REMOTE else "Remote API: off")
        msgs.append("Local model: ✅" if model_ready() else "Local model: ❌")
        df = raw_df()
        msgs.append("Data: ✅" if (df is not None and not df.empty) else "Data: ❌")
        msgs.append(f"S3: {'on' if S3_ENABLED else 'off'}")
        return " | ".join(msgs)

    @reactive.effect
    def _once_sanity():
        if not USE_REMOTE:
            return
        feats = features_df()
        if feats is None or feats.empty:
            return
        sample = feats[FEATURES].head(2)
        try:
            test_probs = remote_predict_proba(sample, FEATURES)
            log.info("sanity.remote.ok n=%s sample=%s", len(test_probs), test_probs[:3].tolist())
        except Exception as e:
            ui.notification_show(f"Remote sanity check failed: {e}", type="error", duration=10)
            log.warning("sanity.remote.fail err=%s", e)

    @output
    @render.plot
    def gg_top_avg():
        agg = agg_players()
        if agg is None or agg.empty:
            return
        df = agg.sort_values(["avg_win_prob","matches"], ascending=[False, False]).head(15).copy()
        df["player"] = pd.Categorical(df["player"], categories=df.sort_values("avg_win_prob")["player"], ordered=True)
        return (ggplot(df, aes(x="player", y="avg_win_prob")) + geom_col() + coord_flip()
                + scale_y_continuous(limits=[0,1]) + labs(x="", y="Avg win prob", title="Top Avg Win Probability") + theme_bw()).draw()

    @output
    @render.plot
    def gg_bottom_avg():
        agg = agg_players()
        if agg is None or agg.empty:
            return
        df = agg.sort_values(["avg_win_prob","matches"], ascending=[True, False]).head(15).copy()
        df["player"] = pd.Categorical(df["player"], categories=df.sort_values("avg_win_prob")["player"], ordered=True)
        return (ggplot(df, aes(x="player", y="avg_win_prob")) + geom_col() + coord_flip()
                + scale_y_continuous(limits=[0,1]) + labs(x="", y="Avg win prob", title="Lowest Avg Win Probability") + theme_bw()).draw()

    @output
    @render.plot
    def gg_most_matches():
        agg = agg_players()
        if agg is None or agg.empty:
            return
        df = agg.sort_values(["matches","avg_win_prob"], ascending=[False, False]).head(15).copy()
        df["player"] = pd.Categorical(df["player"], categories=df.sort_values("matches")["player"], ordered=True)
        return (ggplot(df, aes(x="player", y="matches")) + geom_col() + coord_flip()
                + labs(x="", y="# matches", title="Most Matches") + theme_bw()).draw()

    @output
    @render.table
    def pred_table():
        df = predictions_df()
        if df is None or df.empty:
            return pd.DataFrame()
        show = df.copy()
        show["date"] = pd.to_datetime(show["date"]).dt.date
        show = show.sort_values("date")
        return show[["date","player","opponent","win_prob_player_id","player_id","opponent_id"]]

app = App(app_ui, server)
