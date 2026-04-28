# Monitoring dan Logging

Folder ini disiapkan untuk memenuhi struktur pengumpulan Kriteria 4.

Isi screenshot secara manual setelah sistem berjalan:

- `1.bukti_serving`
- `4.bukti monitoring Prometheus`
- `5.bukti monitoring Grafana`
- `6.bukti alerting Grafana`

File yang dipakai untuk menjalankan sistem:

- `Monitoring dan Logging/prometheus.yml`
- `Monitoring dan Logging/prometheus_exporter.py`
- `Monitoring dan Logging/inference.py`

Saran dashboard metrics untuk dinilai:

- `model_proxy_requests_total`
- `model_proxy_success_total`
- `model_proxy_failures_total`
- `model_proxy_request_latency_seconds`
- `model_proxy_request_payload_bytes`
- `model_proxy_response_payload_bytes`
- `model_proxy_batch_records`
- `model_proxy_health_status`
- `model_proxy_last_prediction_timestamp`
- `model_proxy_prediction_class_total`
- `model_proxy_upstream_status_total`
