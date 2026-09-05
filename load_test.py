"""
load_test.py
-------------
A small, dependency-light load test for POST /recover. Fires N
concurrent workers at the endpoint for a fixed duration and reports
p50/p95/p99 latency, throughput, and how many requests hit the
worker-pool's backpressure (429) — i.e. real numbers to put in the
README instead of a vague "I added async processing" claim.

Usage:
    python load_test.py --url http://localhost:8000/recover \
        --concurrency 50 --duration 30
"""
import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "YesBank", "IDFC"]


def random_payload():
    return {
        "txn_id": f"txn_load_{random.randint(1, 10_000_000)}",
        "amount": round(random.uniform(100, 25000), 2),
        "bank": random.choice(BANKS),
        "hour": random.randint(0, 23),
        "previous_failures": random.choice([0, 0, 0, 1, 1, 2, 3]),
        "is_subscription": random.choice([0, 1]),
        "days_since_last_success": random.randint(1, 75),
    }


def worker(url, stop_at, latencies, status_counts, lock):
    session = requests.Session()
    while time.time() < stop_at:
        payload = random_payload()
        start = time.perf_counter()
        try:
            resp = session.post(url, json=payload, timeout=20)
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = resp.status_code
        except requests.RequestException:
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = -1  # connection-level failure

        with lock:
            latencies.append(elapsed_ms)
            status_counts[status] = status_counts.get(status, 0) + 1


def percentile(data, pct):
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/recover")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds")
    args = parser.parse_args()

    latencies = []
    status_counts = {}
    lock = threading.Lock()
    stop_at = time.time() + args.duration

    print(f"Load testing {args.url} with {args.concurrency} concurrent workers for {args.duration:.0f}s...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(worker, args.url, stop_at, latencies, status_counts, lock)
            for _ in range(args.concurrency)
        ]
        for f in futures:
            f.result()
    wall_time = time.time() - start

    total = len(latencies)
    ok = status_counts.get(200, 0)
    throttled = status_counts.get(429, 0)
    errors = total - ok - throttled

    result = {
        "url": args.url,
        "concurrency": args.concurrency,
        "duration_s": round(wall_time, 2),
        "total_requests": total,
        "requests_per_sec": round(total / wall_time, 2) if wall_time else 0,
        "status_counts": status_counts,
        "ok": ok,
        "throttled_429": throttled,
        "errors": errors,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p90": round(percentile(latencies, 90), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "max": round(max(latencies), 1) if latencies else 0,
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
