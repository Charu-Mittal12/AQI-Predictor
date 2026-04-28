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

## System Architecture
<img width="1408" height="768" alt="Architectural_diagram" src="https://github.com/user-attachments/assets/1738bb21-05a4-4218-a515-6a2eebcb9f65" />


**Key Components:**
- **Streamlit UI (:8501):** Interactive dashboard for city selection and 24-hour AQI forecasts.
- **FastAPI Backend (:8000):** Core REST API managing predictions, health checks, and metrics.
- **MLflow Server (:5001):** Hosts the trained LightGBM model for inference.
- **SQLite DB:** Live data buffer shared between FastAPI and Airflow.
- **Apache Airflow (:8080):** Separate stack for hourly data ingestion from OpenAQ/Open-Meteo.
- **Prometheus/Grafana:** Full observability stack for drift monitoring and system health.

**Data Flow:** OpenAQ → Airflow → SQLite → FastAPI → LightGBM → Streamlit

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
