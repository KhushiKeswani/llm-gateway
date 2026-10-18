
import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

GATEWAY_URL = "http://127.0.0.1:8000/chat"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use identical prompts for each provider.
PROMPTS = [
    "Explain what an API gateway does in two sentences.",
    "What is the difference between HTTP and HTTPS?",
    "Explain caching in simple terms and give one example.",
    "What is a database index, and why is it useful?",
    "Explain the difference between synchronous and asynchronous code.",
]

# Fill these in only with verified pricing for your chosen model/account.
# Leave at zero to calculate no estimated cost.
COST_PER_MILLION_INPUT_TOKENS = {
    "ollama": 0.0,
    "gemini": 0.0,
}
COST_PER_MILLION_OUTPUT_TOKENS = {
    "ollama": 0.0,
    "gemini": 0.0,
}

FIELDS = [
    "timestamp", "provider", "prompt_id", "run", "concurrency",
    "http_status", "success", "ttft_ms", "total_latency_ms",
    "output_chars", "output_text", "error", "quality_score",
    "estimated_cost",
]


def percentile(values, percentile_value):
    """Linear-interpolated percentile."""
    if not values:
        return None

    values = sorted(values)
    position = (len(values) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] + (values[upper] - values[lower]) * fraction


def append_csv(rows, filename):
    path = OUTPUT_DIR / filename
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to: {path}")


def estimate_cost(provider, input_tokens, output_tokens):
    # This estimate is only meaningful if token counts and prices are known.
    input_price = COST_PER_MILLION_INPUT_TOKENS.get(provider, 0.0)
    output_price = COST_PER_MILLION_OUTPUT_TOKENS.get(provider, 0.0)

    if input_price == 0 and output_price == 0:
        return ""

    return (
        input_tokens * input_price
        + output_tokens * output_price
    ) / 1_000_000


def benchmark_one(provider, prompt, prompt_id, run_number, concurrency):
    start = time.perf_counter()
    first_token_time = None
    output_parts = []
    error_message = ""
    http_status = None
    done_received = False

    try:
        with httpx.stream(
            "POST",
            GATEWAY_URL,
            json={"prompt": prompt, "provider": provider},
            timeout=180.0,
        ) as response:
            http_status = response.status_code
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()
                if not payload:
                    continue

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    error_message = "Invalid JSON in SSE event"
                    continue

                event_type = event.get("type")
                text = event.get("text", "")

                if event_type == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_parts.append(text)

                elif event_type == "error":
                    error_message = text or "Gateway error event"

                elif event_type == "done":
                    done_received = True
                    break

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()
    output_text = "".join(output_parts)

    # A request succeeds only if the gateway sends done and no error event.
    success = done_received and not error_message

    ttft_ms = (
        (first_token_time - start) * 1000
        if first_token_time is not None
        else ""
    )

    # End timing after the complete stream has been consumed.
    total_latency_ms = (end - start) * 1000

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "prompt_id": prompt_id,
        "run": run_number,
        "concurrency": concurrency,
        "http_status": http_status if http_status is not None else "",
        "success": success,
        "ttft_ms": round(ttft_ms, 2) if ttft_ms != "" else "",
        "total_latency_ms": round(total_latency_ms, 2),
        "output_chars": len(output_text),
        "output_text": output_text,
        "error": error_message,
        "quality_score": "",
        "estimated_cost": "",
    }


def summarize(rows):
    print("\nSUMMARY")
    print("-" * 72)

    providers = sorted(set(row["provider"] for row in rows))

    for provider in providers:
        subset = [r for r in rows if r["provider"] == provider]
        successful = [r for r in subset if r["success"]]

        ttfts = [
            float(r["ttft_ms"])
            for r in successful
            if r["ttft_ms"] != ""
        ]
        totals = [
            float(r["total_latency_ms"])
            for r in successful
        ]

        success_rate = (
            len(successful) / len(subset) * 100
            if subset else 0
        )

        print(f"\nProvider: {provider}")
        print(f"Requests: {len(subset)}")
        print(f"Successful: {len(successful)}")
        print(f"Success rate: {success_rate:.1f}%")

        if ttfts:
            print(f"Median TTFT: {statistics.median(ttfts):.2f} ms")
            print(f"P95 TTFT: {percentile(ttfts, 0.95):.2f} ms")

        if totals:
            print(f"Median total latency: {statistics.median(totals):.2f} ms")
            print(f"P95 total latency: {percentile(totals, 0.95):.2f} ms")


def run_sequential(providers, repetitions, concurrency=1):
    rows = []

    for provider in providers:
        for prompt_id, prompt in enumerate(PROMPTS, start=1):
            for run_number in range(1, repetitions + 1):
                print(
                    f"Testing {provider}: prompt {prompt_id}, "
                    f"run {run_number}/{repetitions}"
                )

                row = benchmark_one(
                    provider, prompt, prompt_id, run_number, concurrency
                )
                rows.append(row)

                if row["success"]:
                    print(
                        f"  TTFT={row['ttft_ms']} ms, "
                        f"total={row['total_latency_ms']} ms"
                    )
                else:
                    print(f"  FAILED: {row['error']}")

    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    append_csv(rows, filename)
    summarize(rows)
    return rows


async def run_concurrent(providers, repetitions, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    rows = []

    async def one(provider, prompt_id, prompt, run_number):
        async with semaphore:
            return await asyncio.to_thread(
                benchmark_one,
                provider,
                prompt,
                prompt_id,
                run_number,
                concurrency,
            )

    tasks = [
        one(provider, prompt_id, prompt, run_number)
        for provider in providers
        for prompt_id, prompt in enumerate(PROMPTS, start=1)
        for run_number in range(1, repetitions + 1)
    ]

    for task in asyncio.as_completed(tasks):
        row = await task
        rows.append(row)
        print(
            f"{row['provider']}: success={row['success']}, "
            f"TTFT={row['ttft_ms']}, total={row['total_latency_ms']} ms"
        )

    filename = f"concurrency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    append_csv(rows, filename)
    summarize(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["ollama", "gemini", "auto"],
        default=["ollama", "gemini"],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    if args.runs < 1 or args.concurrency < 1:
        parser.error("--runs and --concurrency must be at least 1")

    if args.concurrency == 1:
        run_sequential(args.providers, args.runs)
    else:
        asyncio.run(
            run_concurrent(
                args.providers, args.runs, args.concurrency
            )
        )


if __name__ == "__main__":
    main()
