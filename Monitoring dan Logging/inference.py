import argparse
import csv
import json
import os
import time
import urllib.request


DEFAULT_TARGET_URL = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000/predict")
DEFAULT_DATASET_PATH = os.getenv("INFERENCE_DATASET_PATH", "../MLProject/liver_patient_preprocessing.csv")


def load_sample_records(dataset_path, limit):
    with open(dataset_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            row.pop("Selector", None)
            rows.append({key: float(value) for key, value in row.items()})
            if len(rows) >= limit:
                break
    return rows


def invoke_model(target_url, payload):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        target_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Send sample inference traffic to the monitoring proxy.")
    parser.add_argument("--url", default=DEFAULT_TARGET_URL, help="Prediction URL exposed by prometheus_exporter.py.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to the sample dataset CSV file.")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests to send.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of rows in each request.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests in seconds.")
    args = parser.parse_args()

    records = load_sample_records(args.dataset, args.requests * args.batch_size)
    if not records:
        raise SystemExit("No records found in the dataset.")

    for request_index in range(args.requests):
        start = request_index * args.batch_size
        end = start + args.batch_size
        batch = records[start:end]
        payload = {"dataframe_records": batch}
        status, body = invoke_model(args.url, payload)
        print(f"[{request_index + 1}/{args.requests}] status={status} response={body}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
