# AQI Predictor

> Real-time 24-hour Air Quality Index forecasting for 5 major Indian cities using LightGBM, OpenAQ v3, Apache Airflow, MLflow, FastAPI, and Streamlit.

---

## Overview

AQI Predictor is a production-grade, end-to-end MLOps system that:
- **Ingests:** Live pollutant data from **OpenAQ v3** and weather from **Open-Meteo**.
- **Processes:** 11-stage automated DVC pipeline for feature engineering and training.
- **Predicts:** Multi-output **LightGBM** forecasts for the next 24 hours.
- **Monitors:** Real-time **data drift** (PSI, KS-test, Z-score) and system health.
- **Visualizes:** Interactive Streamlit dashboard with 7-day trends and health advisories.
- **Observability:** Prometheus + Grafana stack for metrics and alerts.

---

## Architecture

The system uses a modular deployment:
1. **Application Stack:** FastAPI, Streamlit, MLflow, and Monitoring services.
2. **Orchestration Stack:** Apache Airflow (separate containerized stack) managing hourly ingestion, cleanup, and drift detection.

```text
OpenAQ v3 API + Open-Meteo ──► Airflow DAGs (Ingestion/Drift) ───► SQLite DB
                                                                    │
   Streamlit ──► FastAPI Backend ───► MLflow Server ───► LightGBM Model
   Frontend         │                                           │
                    └──► Prometheus ───► Grafana ───────────────┘
```

---

## Project Structure

```text
AQIPredictor/
├── backend/            # FastAPI app, model service, DB services, & tests
├── airflow/            # Separate Airflow stack (DAGs, Docker, logs)
├── frontend/           # Streamlit dashboard
├── mlflow/             # MLflow artifact store
├── monitoring/         # Prometheus configs & AlertManager rules
├── scripts/            # Training, EDA, and data preparation scripts
├── dvc.yaml            # 11-stage DVC pipeline
├── docker-compose.yml  # Application stack definition
└── README.md
```

---

## Running the Project

### 1. Prerequisites
Ensure you have **Docker** and **Docker Compose** installed.

### 2. Setup
```bash
git clone <your-repo-url>
cd AQI_Predictor
cp .env.example .env
# Edit .env and configure your OPENAQ_API_KEY and other parameters
```

### 3. Deploy Application Stack
Start the backend, frontend, MLflow, and monitoring services:
```bash
docker-compose up --build -d
```

### 4. Deploy Airflow Stack
Airflow runs in a separate containerized stack to manage your data orchestration.
```bash
cd airflow
# Download the official Airflow image
docker pull apache/airflow:latest
# Start the Airflow stack
docker-compose up -d
```

### 5. Verify & Access
- **FastAPI/Docs:** `http://localhost:8000/docs`
- **Frontend:** `http://localhost:8501`
- **Airflow:** `http://localhost:8080` (User: airflow/airflow)
- **Grafana:** `http://localhost:3000`

**Trigger Ingestion:**
```bash
curl -X POST http://localhost:8000/admin/sync
```

**Get Prediction:**
```bash
curl "http://localhost:8000/predict?city=Chennai&location_id=3135"
```

---

## Main API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/ready` | MLflow/model readiness check |
| GET | `/predict?city=&location_id=` | 24-hour AQI forecast |
| GET | `/cities` | List supported cities |
| GET | `/metrics` | Prometheus metrics scrape |
| POST | `/admin/sync` | Trigger ingestion manually |

---

## Testing

```bash
cd backend
pytest tests/ -v --tb=short
```

The test suite covers health endpoints, prediction logic, model info, database operations, advisory logic, and drift detection. The project includes 47 backend-focused test cases.

---

## Drift Detection & Monitoring

Drift detection runs hourly via the Airflow DAG. It compares recent live data against older live data using:
- **PSI** (Population Stability Index) for distribution shift.
- **KS-test** for statistical significance.
- **Z-score** for mean shift.

Results are written to the `drift_results` SQLite table and exposed to Prometheus via a custom `DriftCollector`.

---

## License

MIT License.
