# mlflow/Dockerfile
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir mlflow lightgbm scikit-learn pandas numpy

EXPOSE 5001

CMD ["mlflow", "models", "serve", \
     "-m", "models:/AQI_Model/Production", \
     "-p", "5001", "--host", "0.0.0.0", "--no-conda"]