# Monitoring and Logging Guide

This repo now includes a minimal monitoring stack for Kriteria 4:

- `mlflow models serve --no-conda` serves the logged model artifact.
- `Monitoring dan Logging/prometheus_exporter.py` exposes `/predict`, `/health`, and `/metrics`.
- `Monitoring dan Logging/prometheus.yml` scrapes the monitoring proxy.
- `docker-compose.monitoring.yml` runs Prometheus and Grafana.
- `Monitoring dan Logging/inference.py` generates traffic so metrics appear in Prometheus and Grafana.

## Run Order

1. Serve the model from MLflow:

   ```bash
   mlflow models serve -m "runs:/YOUR_RUN_ID/model" -p 5001 --no-conda
   ```

2. Install the metrics dependency:

   ```bash
   pip install prometheus-client
   ```

3. Start the monitoring proxy:

   ```bash
   python "Monitoring dan Logging/prometheus_exporter.py"
   ```

4. Generate sample traffic:

   ```bash
   python "Monitoring dan Logging/inference.py" --requests 10 --batch-size 1
   ```

5. Start Prometheus and Grafana:

   ```bash
   docker compose -f docker-compose.monitoring.yml up -d
   ```

6. Open the services:

   ```text
   Model proxy: http://127.0.0.1:8000
   Prometheus: http://127.0.0.1:9090
   Grafana: http://127.0.0.1:3000
   ```

## Useful Endpoints

- `GET /health`
- `GET /metrics`
- `POST /predict`

## Available Metrics

- `model_proxy_requests_total`
- `model_proxy_success_total`
- `model_proxy_failures_total`
- `model_proxy_upstream_status_total`
- `model_proxy_request_latency_seconds`
- `model_proxy_request_payload_bytes`
- `model_proxy_response_payload_bytes`
- `model_proxy_batch_records`
- `model_proxy_health_status`
- `model_proxy_last_prediction_timestamp`
- `model_proxy_prediction_class_total`

## Example Inference Payload

```json
{
  "dataframe_records": [
    {
      "Age": 65,
      "Gender": 1,
      "Total_Bilirubin": 1.0,
      "Direct_Bilirubin": 0.3,
      "Alkaline_Phosphotase": 200,
      "Alamine_Aminotransferase": 30,
      "Aspartate_Aminotransferase": 35,
      "Total_Protiens": 6.5,
      "Albumin": 3.2,
      "Albumin_and_Globulin_Ratio": 0.9
    }
  ]
}
```
