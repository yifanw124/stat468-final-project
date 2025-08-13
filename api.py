# api.py — robust load from S3 for vetiver + stable JSON endpoints
import os
from io import BytesIO
from typing import Any, List, Dict

import boto3
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, Body, HTTPException

S3_BUCKET  = os.getenv("MODEL_BUCKET", "badminton12345")
MODEL_KEY  = os.getenv("MODEL_KEY", "stack_model/vetiver_model.joblib")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ALLOWED    = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

FEATURES = [
    "elo_diff",
    "win_pct_5", "win_pct_10", "win_pct_20",
    "h2h_decay", "h2h_adj",
    "pr_player", "pr_opponent",
]

# ---- boto3 client ----
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
else:
    s3 = boto3.client("s3", region_name=AWS_REGION)

# ---- download & load ----
print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY} ...")
buf = BytesIO()
s3.download_fileobj(S3_BUCKET, MODEL_KEY, buf)
buf.seek(0)
loaded: Any = joblib.load(buf)

# ---- build Vetiver app no matter what was saved ----
from vetiver import VetiverModel, VetiverAPI

def make_vetiver_model(obj: Any) -> VetiverModel:
    # Case A: already a VetiverModel -> use as-is
    if isinstance(obj, VetiverModel):
        return obj

    # Case B: dict bundle -> pull estimator out
    if isinstance(obj, dict) and "model" in obj:
        estimator = obj["model"]
    else:
        estimator = obj  # plain sklearn estimator

    # Minimal prototype ensures schema
    prototype = pd.DataFrame([{c: 0.0 for c in FEATURES}])
    return VetiverModel(model=estimator, model_name="stack_model", prototype=prototype)

v = make_vetiver_model(loaded)
# Relax prototype checks to avoid strict pydantic schema issues
v_api = VetiverAPI(v, check_prototype=False)
app = v_api.app

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Stable custom endpoints: accept list[dict] ----
router = APIRouter()

def _to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=422, detail="Body must be a non-empty JSON list")
    X = pd.DataFrame(records)
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    X = X[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

@router.post("/predict_json")
def predict_json(records: List[Dict[str, Any]] = Body(...)):
    X = _to_df(records)
    try:
        y = v.model.predict(X)
        # coerce numpy types to py types
        return {"predictions": [int(getattr(x, "item", lambda: x)()) if hasattr(x, "item") else int(x) for x in y]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict failed: {e}")

@router.post("/predict_proba")
def predict_proba(records: List[Dict[str, Any]] = Body(...)):
    X = _to_df(records)
    try:
        if hasattr(v.model, "predict_proba"):
            proba = v.model.predict_proba(X)
            if getattr(proba, "ndim", 1) == 2 and proba.shape[1] == 2:
                return {"probabilities": proba[:, 1].tolist()}
            return {"probabilities": proba.tolist()}
        raise HTTPException(status_code=400, detail="Model has no predict_proba()")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_proba failed: {e}")

@router.get("/hello")
def hello():
    return {"ok": True, "msg": "Vetiver API running. Try /ping, /predict_json, /predict_proba, /metadata."}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
