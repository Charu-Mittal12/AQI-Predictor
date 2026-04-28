# AQI Predictor

> Real-time 24-hour Air Quality Index forecasting for 5 major Indian cities using LightGBM, OpenAQ v3, Apache Airflow, MLflow, FastAPI, and Streamlit.

---

## Overview

AQI Predictor is an end-to-end MLOps project that:
- Ingests live air-quality sensor data from **OpenAQ v3**
- Enriches data with **weather features**
- Detects **data drift** using PSI, KS-test, and Z-score
- Serves **24-hour AQI forecasts** through a FastAPI backend
- Displays predictions, pollutant levels, and advisories in a Streamlit dashboard
- Monitors the system with **Prometheus** and **Grafana** [file:3]

---

## Supported Cities

- Chennai
- Delhi
- Mumbai
- Kolkata
- Ahmedabad [file:3]

---

## Key Features

- **24-hour AQI prediction** for selected city and station [file:3]
- **Hourly ingestion pipeline** using Apache Airflow [file:3]
- **SQLite live buffer** for recent sensor and prediction data [file:3]
- **MLflow model serving** for LightGBM inference on port 5001 [file:3]
- **Drift monitoring** written to `driftresults` and exposed via Prometheus metrics [file:3]
- **Health, readiness, metrics, and admin APIs** for operations support [file:3]

---

## Architecture

```text
OpenAQ v3 API + Open-Meteo
          │
          ▼
   Airflow DAGs
(bootstrap → ingest → cleanup → drift)
          │
          ▼
      SQLite DB
          │
          ▼
   FastAPI Backend  ───► MLflow Server ───► LightGBM Model
          │
          ├──► Streamlit Frontend
          └──► Prometheus ───► Grafana
```

This system uses Airflow for scheduled ingestion and cleanup, FastAPI for prediction APIs, MLflow for model serving, and Streamlit for user interaction. Prometheus scrapes metrics from the backend, while Grafana visualizes system and drift health. [file:3]

---

## Project Structure

```text
AQIPredictor/
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   ├── services/
│   │   ├── schemas/
│   │   └── utils/
│   ├── deployment_bundle/
│   ├── drift_baseline/
│   ├── database/
│   ├── Dockerfile
│   └── requirements.txt
├── airflow/
│   └── dags/
├── frontend/
├── mlflow/
├── monitoring/
├── tests/
├── docker-compose.yml
├── scripts/    #files used for training
└── dvc.yaml
└── params.yaml
└── eda.py
└── README.md

```

The backend contains the FastAPI application, model logic, data services, and monitoring utilities. Airflow DAGs manage ingestion and cleanup, while the frontend provides the Streamlit dashboard. [file:3]

---

## Main API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check [file:3] |
| GET | `/ready` | MLflow/model readiness check [file:3] |
| GET | `/cities` | List supported cities [file:3] |
| GET | `/predict?city=&location_id=` | 24-hour AQI forecast [file:3] |
| GET | `/model-info` | Model metadata [file:3] |
| GET | `/metrics` | Prometheus metrics scrape [file:3] |
| POST | `/admin/sync` | Trigger ingestion manually [file:3] |
| POST | `/admin/prune` | Prune old hourly measurements [file:3] |

Interactive Swagger docs are available at `http://localhost:8000/docs` when the backend is running. [file:3]

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic, Uvicorn [file:3] |
| Model | LightGBM, scikit-learn [file:3] |
| Model Serving | MLflow [file:3] |
| Orchestration | Apache Airflow [file:3] |
| Database | SQLite [file:3] |
| Frontend | Streamlit, Plotly [file:3] |
| Monitoring | Prometheus, Grafana [file:3] |
| Testing | pytest, FastAPI TestClient [file:3] |
| Containerization | Docker, Docker Compose [file:3] |

---

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AQIPredictor
```

### 2. Create environment file

```bash
cp .env.example .env
```

Add your OpenAQ API key and any required paths in `.env`. The project reads variables such as `OPENAQ_API_KEY`, `DB_PATH`, `MODEL_NAME`, `MODEL_STAGE`, and `LIVE_HISTORY_HOURS`. [file:3]

### 3. Start all services

```bash
docker-compose up --build
```

Expected services include:
- FastAPI backend on **8000**
- Streamlit frontend on **8501**
- MLflow model server on **5001**
- Airflow webserver on **8080**
- Prometheus on **9090**
- Grafana on **3000** [file:3]

### 4. Trigger ingestion

```bash
curl -X POST http://localhost:8000/admin/sync
```

### 5. Request a prediction

```bash
curl "http://localhost:8000/predict?city=Chennai&location_id=3135"
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v --tb=short
```

The test suite covers health endpoints, prediction logic, model info, database operations, advisory logic, and drift detection. Your project contains 47 backend-focused test cases based on the provided test plan and test files. [file:3]

---

## Drift Detection

Drift detection runs after ingestion in the Airflow DAG and compares recent live data against older live data from the same SQLite source. It uses:
- **PSI** for distribution shift
- **KS-test** for statistical significance
- **Z-score** for mean shift [file:3]

Results are written to the `driftresults` table and exposed through Prometheus metrics using a custom `DriftCollector`. [file:3]

---

## Monitoring

Prometheus metrics include:
- Prediction request count
- Prediction latency
- Live fetch request count
- Live fetch rows written
- Latest DB age
- Model readiness
- Drift PSI / drift detected / drifted feature count [file:3]

Grafana can be used to visualize backend performance, drift signals, and freshness metrics. [file:3]

---

## License

MIT License.
