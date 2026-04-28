FROM python:3.13-slim

WORKDIR /backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/backend \
    ENABLE_STARTUP_BACKFILL=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /backend/requirements.txt

COPY app /backend/app
COPY deployment_bundle /backend/deployment_bundle
COPY drift_baseline /backend/drift_baseline 

RUN mkdir -p /backend/database

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
