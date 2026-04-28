import json
import os
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


MODEL_INVOCATIONS_URL = os.getenv("MODEL_INVOCATIONS_URL", "http://127.0.0.1:5001/invocations")
MODEL_PING_URL = os.getenv("MODEL_PING_URL", "http://127.0.0.1:5001/ping")
EXPORTER_HOST = os.getenv("EXPORTER_HOST", "0.0.0.0")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))

REQUESTS_TOTAL = Counter(
    "model_proxy_requests_total",
    "Total requests handled by the monitoring proxy.",
    ["endpoint", "method", "status"],
)
SUCCESS_TOTAL = Counter(
    "model_proxy_success_total",
    "Successful prediction requests handled by the monitoring proxy.",
)
FAILURES_TOTAL = Counter(
    "model_proxy_failures_total",
    "Failed requests handled by the monitoring proxy.",
    ["reason"],
)
UPSTREAM_STATUS_TOTAL = Counter(
    "model_proxy_upstream_status_total",
    "Upstream MLflow model server status codes.",
    ["status_code"],
)
REQUEST_LATENCY_SECONDS = Histogram(
    "model_proxy_request_latency_seconds",
    "Prediction request latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
REQUEST_PAYLOAD_BYTES = Histogram(
    "model_proxy_request_payload_bytes",
    "Request payload size in bytes.",
    buckets=(128, 512, 1024, 4096, 16384, 65536),
)
RESPONSE_PAYLOAD_BYTES = Histogram(
    "model_proxy_response_payload_bytes",
    "Response payload size in bytes.",
    buckets=(128, 512, 1024, 4096, 16384, 65536),
)
BATCH_RECORDS = Histogram(
    "model_proxy_batch_records",
    "Number of records sent in a prediction request.",
    buckets=(1, 2, 5, 10, 20, 50, 100),
)
HEALTH_STATUS = Gauge(
    "model_proxy_health_status",
    "Health status of the upstream MLflow model server. 1 means healthy, 0 means unhealthy.",
)
LAST_PREDICTION_TIMESTAMP = Gauge(
    "model_proxy_last_prediction_timestamp",
    "Unix timestamp of the last successful prediction.",
)
PREDICTION_CLASS_TOTAL = Counter(
    "model_proxy_prediction_class_total",
    "Total predicted classes returned by the model.",
    ["prediction"],
)


def _extract_record_count(payload):
    if not isinstance(payload, dict):
        return 0
    for key in ("dataframe_records", "instances", "inputs"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _extract_predictions(response_json):
    if isinstance(response_json, dict):
        for key in ("predictions", "outputs", "data"):
            value = response_json.get(key)
            if isinstance(value, list):
                return value
    if isinstance(response_json, list):
        return response_json
    return []


def _check_upstream_health():
    request = urllib.request.Request(MODEL_PING_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=MODEL_TIMEOUT_SECONDS) as response:
            healthy = 1 if response.status == HTTPStatus.OK else 0
            HEALTH_STATUS.set(healthy)
            return healthy, response.status
    except Exception:
        HEALTH_STATUS.set(0)
        return 0, None


class MonitoringHandler(BaseHTTPRequestHandler):
    server_version = "ModelMonitoringProxy/1.0"

    def _write_response(self, status_code, body, content_type="application/json"):
        body_bytes = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self):
        if self.path == "/":
            body = json.dumps(
                {
                    "message": "Model monitoring proxy is running.",
                    "predict_url": "/predict",
                    "metrics_url": "/metrics",
                    "health_url": "/health",
                }
            )
            REQUESTS_TOTAL.labels(endpoint="/", method="GET", status="200").inc()
            self._write_response(HTTPStatus.OK, body)
            return

        if self.path == "/metrics":
            REQUESTS_TOTAL.labels(endpoint="/metrics", method="GET", status="200").inc()
            self._write_response(HTTPStatus.OK, generate_latest(), CONTENT_TYPE_LATEST)
            return

        if self.path == "/health":
            healthy, upstream_status = _check_upstream_health()
            status_label = str(upstream_status or 503)
            REQUESTS_TOTAL.labels(endpoint="/health", method="GET", status=status_label).inc()
            if upstream_status is not None:
                UPSTREAM_STATUS_TOTAL.labels(status_code=str(upstream_status)).inc()
            status_code = HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE
            body = json.dumps(
                {
                    "healthy": bool(healthy),
                    "upstream_status": upstream_status,
                    "model_ping_url": MODEL_PING_URL,
                }
            )
            self._write_response(status_code, body)
            return

        REQUESTS_TOTAL.labels(endpoint="unknown", method="GET", status="404").inc()
        self._write_response(HTTPStatus.NOT_FOUND, json.dumps({"error": "Not found"}))

    def do_POST(self):
        if self.path != "/predict":
            REQUESTS_TOTAL.labels(endpoint="unknown", method="POST", status="404").inc()
            self._write_response(HTTPStatus.NOT_FOUND, json.dumps({"error": "Not found"}))
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload_bytes = self.rfile.read(content_length)
        REQUEST_PAYLOAD_BYTES.observe(len(payload_bytes))
        started_at = time.time()

        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            record_count = _extract_record_count(payload)
            if record_count:
                BATCH_RECORDS.observe(record_count)
        except json.JSONDecodeError:
            FAILURES_TOTAL.labels(reason="invalid_json").inc()
            REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status="400").inc()
            self._write_response(HTTPStatus.BAD_REQUEST, json.dumps({"error": "Invalid JSON payload"}))
            return

        request = urllib.request.Request(
            MODEL_INVOCATIONS_URL,
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=MODEL_TIMEOUT_SECONDS) as response:
                response_bytes = response.read()
                duration = time.time() - started_at
                REQUEST_LATENCY_SECONDS.observe(duration)
                RESPONSE_PAYLOAD_BYTES.observe(len(response_bytes))
                REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status=str(response.status)).inc()
                UPSTREAM_STATUS_TOTAL.labels(status_code=str(response.status)).inc()
                SUCCESS_TOTAL.inc()
                LAST_PREDICTION_TIMESTAMP.set(time.time())

                try:
                    predictions = _extract_predictions(json.loads(response_bytes.decode("utf-8")))
                    for prediction in predictions:
                        PREDICTION_CLASS_TOTAL.labels(prediction=str(prediction)).inc()
                except json.JSONDecodeError:
                    FAILURES_TOTAL.labels(reason="invalid_upstream_json").inc()

                self._write_response(response.status, response_bytes)
        except urllib.error.HTTPError as error:
            duration = time.time() - started_at
            REQUEST_LATENCY_SECONDS.observe(duration)
            body = error.read()
            RESPONSE_PAYLOAD_BYTES.observe(len(body))
            FAILURES_TOTAL.labels(reason="upstream_http_error").inc()
            REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status=str(error.code)).inc()
            UPSTREAM_STATUS_TOTAL.labels(status_code=str(error.code)).inc()
            self._write_response(error.code, body)
        except Exception as error:
            duration = time.time() - started_at
            REQUEST_LATENCY_SECONDS.observe(duration)
            FAILURES_TOTAL.labels(reason="upstream_connection_error").inc()
            REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status="502").inc()
            self._write_response(
                HTTPStatus.BAD_GATEWAY,
                json.dumps(
                    {
                        "error": "Failed to connect to the MLflow model server.",
                        "details": str(error),
                    }
                ),
            )

    def log_message(self, format, *args):
        return


def main():
    _check_upstream_health()
    server = ThreadingHTTPServer((EXPORTER_HOST, EXPORTER_PORT), MonitoringHandler)
    print(
        f"Monitoring proxy listening on http://{EXPORTER_HOST}:{EXPORTER_PORT} "
        f"and forwarding predictions to {MODEL_INVOCATIONS_URL}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
