# api.py — Vetiver API loading from S3 (supports VetiverModel or plain sklearn model)
import os
from io import BytesIO
from typing import Any

import boto3
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ---- config ----
S3_BUCKET  = os.getenv("MODEL_BUCKET", "badminton12345")
MODEL_KEY  = os.getenv("MODEL_KEY", "stack_model/vetiver_model.joblib")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ALLOWED    = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Training feature order (schema Vetiver will enforce)
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

# ---- download & load model object ----
print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY} ...")
buf = BytesIO()
s3.download_fileobj(S3_BUCKET, MODEL_KEY, buf)
buf.seek(0)
obj: Any = joblib.load(buf)

# ---- build Vetiver app ----
from vetiver import VetiverModel, VetiverAPI

# Explicit prototype (1-row DataFrame with the correct column names/dtypes)
prototype = pd.DataFrame([{c: 0.0 for c in FEATURES}])

# If the artifact already is a VetiverModel, use it directly; else wrap a plain sklearn model
if isinstance(obj, VetiverModel):
    v = obj
else:
    # If you saved a dict bundle, pull the estimator
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    v = VetiverModel(model=model, model_name="stack_model", prototype=prototype)

# Enforce schema to avoid the “1D array” error
v_api = VetiverAPI(v, check_prototype=True)
app = v_api.app

# CORS (so Shiny/browser can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Optional hello route
try:
    from fastapi import APIRouter
    router = APIRouter()

    @router.get("/hello")
    def hello():
        return {"ok": True, "msg": "Vetiver API running. Try /ping, /predict, /metadata."}

    app.include_router(router)
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)