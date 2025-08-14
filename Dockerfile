cat > Dockerfile <<'EOF'
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    AWS_DEFAULT_REGION=us-east-1
WORKDIR /app
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN python -V && pip -V \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN python - << 'PY'
import importlib, sys
for pkg in ["boto3","vetiver","fastapi","uvicorn","pandas","numpy","scikit-learn","xgboost","networkx","joblib"]:
    try:
        importlib.import_module(pkg.replace("-","_"))
    except Exception as e:
        sys.exit(f"Missing {pkg}: {e}")
print("All deps import OK.")
PY
COPY api.py /app/
ENV MODEL_BUCKET=badminton12345
ENV MODEL_KEY=stack_model/vetiver_model.joblib
ENV ALLOWED_ORIGINS=*
EXPOSE 8000
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000"]