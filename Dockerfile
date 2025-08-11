FROM python:3.11-slim
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# copy only the API/shiny code (adjust path if your app is elsewhere)
COPY model/app.py /app/app.py
ENV MODEL_BUCKET=badminton12345 \
    MODEL_KEY=stack_model.pkl \
    DATA_KEY=tournaments_2018_2025_June.csv \
    AWS_REGION=us-east-1
EXPOSE 8080
CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8080"]
