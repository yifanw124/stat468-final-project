# api.py — Vetiver API with schema prototype so /predict accepts list-of-dicts
import os
from io import BytesIO
from typing import Any

import boto3
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ---- config ----
S3_BUCKET   = os.getenv("MODEL_BUCKET", "badminton12345")
MODEL_KEY   = os.getenv("MODEL_KEY", "stack_model/vetiver_model.joblib")
AWS_REGION  = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ALLOWED     = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

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

# ---- download & load object ----
print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY} ...")
buf = BytesIO()
s3.download_fileobj(S3_BUCKET, MODEL_KEY, buf)
buf.seek(0)
obj: Any = joblib.load(buf)

# Pull estimator if a dict bundle
model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

# Training feature order (must match your train script)
FEATURES = [
    "elo_diff",
    "win_pct_5", "win_pct_10", "win_pct_20",
    "h2h_decay", "h2h_adj",
    "pr_player", "pr_opponent",
]

# Build a 1-row prototype DataFrame with the right columns/dtypes
prototype_df = pd.DataFrame([{k: 0.0 for k in FEATURES}])

# ---- Vetiver app with prototype ----
app = None
try:
    # Newer vetiver API
    from vetiver import VetiverModel, vetiver_api

    v = VetiverModel(
        model=model,
        model_name="stack_model",
        prototype=prototype_df,         # <-- key: tells vetiver how to parse JSON
    )
    app = vetiver_api(v, check_prototype=True)

except ImportError:
    # Older vetiver API
    from vetiver import VetiverModel, VetiverAPI  # type: ignore

    v = VetiverModel(
        model=model,
        model_name="stack_model",
        prototype=prototype_df,
    )
    v_api = VetiverAPI(v, check_prototype=True)
    app = getattr(v_api, "app", v_api)

# ---- CORS ----
try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# Optional hello routev
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
