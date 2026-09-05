"""
chaos_test_idempotency.py
----------------------------
Fires the SAME txn_id at POST /recover N times concurrently to prove
the idempotency guarantee documented in execution/doer.py: no matter
how many times the same failed-payment webhook gets redelivered
within the idempotency window, exactly ONE audit row should be
written and ONE execution should occur — every other call should come
back with duplicate: true.

Usage:
    python chaos_test_idempotency.py --url http://localhost:8000/recover --n 50
"""
import argparse
import concurrent.futures
import json
import requests


def fire_once(url, payload):
    try:
        resp = requests.post(url, json=payload, timeout=15)
        return resp.status_code, (resp.json() if resp.ok else None)
    except requests.RequestException as e:
        return -1, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/recover")
    parser.add_argument("--n", type=int, default=50, help="concurrent duplicate deliveries")
    parser.add_argument("--txn-id", default="txn_chaos_dup_001")
    args = parser.parse_args()

    payload = {
        "txn_id": args.txn_id,
        "amount": 2450.0,
        "bank": "HDFC",
        "hour": 14,
        "previous_failures": 0,
        "is_subscription": 0,
        "days_since_last_success": 5,
    }

    print(f"Firing {args.n} concurrent duplicate deliveries of txn_id={args.txn_id} at {args.url} ...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
        futures = [pool.submit(fire_once, args.url, payload) for _ in range(args.n)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    ok = [r for status, r in results if status == 200 and r]
    duplicates = sum(1 for r in ok if r.get("audit_record", {}).get("duplicate") is True)
    originals = sum(1 for r in ok if r.get("audit_record", {}).get("duplicate") is False)
    errors = len(results) - len(ok)

    print(json.dumps({
        "total_requests": args.n,
        "successful_responses": len(ok),
        "originals_written": originals,
        "duplicates_detected": duplicates,
        "errors": errors,
        "PASS": originals == 1 and duplicates == len(ok) - 1,
    }, indent=2))


if __name__ == "__main__":
    main()