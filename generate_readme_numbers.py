"""
generate_readme_numbers.py
-----------------------------
Runs batch_eval.py and (optionally) load_test.py against a live server,
and prints ready-to-paste Markdown for the two "X%/Y%/Z%/W%" and
load-test placeholders in README.md. This exists purely to remove the
friction of manually copying numbers out of stdout and reformatting
them — it does not fabricate anything; every number comes straight
from your own run.

Usage:
    # 1. Make sure outputs/detective_model.joblib exists:
    python generate_data.py && python train_detective.py

    # 2. Just batch_eval numbers (no server needed):
    python generate_readme_numbers.py

    # 3. Also include load_test numbers (needs a running server):
    uvicorn main:app --reload &
    python generate_readme_numbers.py --load-test --url http://localhost:8000/recover
"""
import argparse
import io
import json
import re
import subprocess
import sys


def run_batch_eval():
    result = subprocess.run(
        [sys.executable, "batch_eval.py"], capture_output=True, text=True
    )
    out = result.stdout
    if result.returncode != 0:
        print("batch_eval.py failed:\n", result.stderr, file=sys.stderr)
        return None

    def grab(pattern):
        m = re.search(pattern, out)
        return m.group(1) if m else None

    total_at_risk = grab(r"Total money at risk\s*:\s*Rs ([\d,]+)")
    do_nothing_pct = grab(r"Baseline - do nothing\s*:.*?\(([\d.]+)% recovered\)")
    retry_all_pct = grab(r"Baseline - retry everything\s*:.*?\(([\d.]+)% recovered\)")
    system_pct = grab(r"OUR SYSTEM\s*:.*?\(([\d.]+)% recovered\)")
    stop_line = grab(r"EXAMPLE: system correctly STOPS.*?\n-+\n(.*?)\n")
    escalate_line = grab(r"EXAMPLE: system ESCALATES.*?\n-+\n(.*?)\n")

    return {
        "raw_output": out,
        "total_at_risk": total_at_risk,
        "do_nothing_pct": do_nothing_pct,
        "retry_all_pct": retry_all_pct,
        "system_pct": system_pct,
        "stop_example": stop_line,
        "escalate_example": escalate_line,
    }


def run_load_test(url, concurrency, duration):
    result = subprocess.run(
        [sys.executable, "load_test.py", "--url", url,
         "--concurrency", str(concurrency), "--duration", str(duration)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("load_test.py failed:\n", result.stderr, file=sys.stderr)
        return None
    # load_test.py prints a JSON blob as its last output block
    try:
        json_start = result.stdout.index("{")
        return json.loads(result.stdout[json_start:])
    except Exception:
        print("Could not parse load_test.py output:\n", result.stdout, file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-test", action="store_true", help="Also run load_test.py (needs a live server)")
    parser.add_argument("--url", default="http://localhost:8000/recover")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    print("Running batch_eval.py ...", file=sys.stderr)
    eval_result = run_batch_eval()

    print("\n" + "=" * 70)
    print("PASTE INTO README — top-of-file business sentence:")
    print("=" * 70)
    if eval_result and eval_result["system_pct"] and eval_result["retry_all_pct"] and eval_result["do_nothing_pct"]:
        print(
            f'the agent recovers **{eval_result["system_pct"]}%** of failed-payment value vs '
            f'**{eval_result["retry_all_pct"]}%** for a naive "retry everything" baseline and '
            f'**{eval_result["do_nothing_pct"]}%** for "do nothing"'
        )
    else:
        print("(Could not auto-extract — check batch_eval.py's raw output below and fill in manually.)")

    if eval_result:
        print("\n--- Full batch_eval.py output (for reference / the STOP & ESCALATE examples) ---")
        print(eval_result["raw_output"])

    if args.load_test:
        print("\nRunning load_test.py ...", file=sys.stderr)
        load_result = run_load_test(args.url, args.concurrency, args.duration)
        print("\n" + "=" * 70)
        print("PASTE INTO README — Reliability & Load Testing section:")
        print("=" * 70)
        if load_result:
            print("```json")
            print(json.dumps(load_result, indent=2))
            print("```")
            lat = load_result.get("latency_ms", {})
            print(
                f'\n**Takeaway:** sustained {load_result.get("requests_per_sec", "?")} req/s '
                f'at p95 latency of {lat.get("p95", "?")} ms with '
                f'{load_result.get("errors", "?")} errors; '
                f'{load_result.get("throttled_429", 0)} requests hit backpressure (429).'
            )
        else:
            print("(load_test.py did not produce parseable output — run it manually.)")


if __name__ == "__main__":
    main()