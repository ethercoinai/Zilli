#!/usr/bin/env python3
"""Zilli vs direct Ollama API benchmark.

Usage:
    python scripts/benchmark.py [--zilli-url http://127.0.0.1:8900] [--ollama-url http://127.0.0.1:11434]
                                [--model qwen3:7b] [--requests 20] [--concurrency 4]
"""

import argparse
import asyncio
import time
from statistics import mean, median, stdev

try:
    import httpx
except ImportError:
    print("httpx required. pip install httpx")
    raise SystemExit(1)

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list of integers.",
    "What is the difference between AI and machine learning?",
    "Summarize the plot of Romeo and Juliet.",
    "Design a simple REST API for a todo list application.",
    "Explain the concept of recursion with an example.",
    "What are the main causes of climate change?",
    "Write a SQL query to find duplicate email addresses.",
    "Describe the process of photosynthesis.",
    "Create a CSS grid layout for a responsive dashboard.",
    "What is the difference between HTTP and HTTPS?",
    "Write a bash script to backup a directory.",
    "Explain the CAP theorem in distributed systems.",
    "How does garbage collection work in Python?",
    "Design a database schema for an e-commerce platform.",
    "What are design patterns? Give three examples.",
    "Write a Dockerfile for a Node.js application.",
    "Explain how a blockchain works.",
    "What is the prisoner's dilemma?",
]

RESULTS: dict[str, list[float]] = {"zilli": [], "ollama": [], "zilli_stream": []}


async def benchmark_zilli(client: httpx.AsyncClient, url: str, model: str, sem: asyncio.Semaphore) -> list[float]:
    latencies = []
    for prompt in PROMPTS:
        async with sem:
            start = time.monotonic()
            try:
                resp = await client.post(
                    f"{url}/v1/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                    timeout=120,
                )
                elapsed = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    latencies.append(elapsed)
                    print(f"  ZILLI  {elapsed:7.0f}ms  {len(content):5d} chars")
                else:
                    print(f"  ZILLI  ERROR {resp.status_code}: {resp.text[:80]}")
            except Exception as e:
                print(f"  ZILLI  ERROR {e}")
    return latencies


async def benchmark_zilli_stream(client: httpx.AsyncClient, url: str, model: str, sem: asyncio.Semaphore) -> list[float]:
    ttft_latencies = []
    for prompt in PROMPTS:
        async with sem:
            start = time.monotonic()
            first_chunk = None
            try:
                async with client.stream(
                    "POST", f"{url}/v1/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True},
                    timeout=120,
                ) as resp:
                    full_text = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            if first_chunk is None:
                                first_chunk = time.monotonic()
                            import json
                            try:
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                full_text += delta
                            except json.JSONDecodeError:
                                pass
                ttft = (first_chunk - start) * 1000 if first_chunk else 0
                total = (time.monotonic() - start) * 1000
                ttft_latencies.append(ttft)
                print(f"  STREAM {ttft:7.0f}ms TTFT  {total:7.0f}ms total  {len(full_text):5d} chars")
            except Exception as e:
                print(f"  STREAM ERROR {e}")
    return ttft_latencies


async def benchmark_ollama_direct(client: httpx.AsyncClient, url: str, model: str, sem: asyncio.Semaphore) -> list[float]:
    latencies = []
    for prompt in PROMPTS:
        async with sem:
            start = time.monotonic()
            try:
                resp = await client.post(
                    f"{url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                elapsed = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("response", "")
                    latencies.append(elapsed)
                    print(f"  OLLAMA {elapsed:7.0f}ms  {len(content):5d} chars")
                else:
                    print(f"  OLLAMA ERROR {resp.status_code}: {resp.text[:80]}")
            except Exception as e:
                print(f"  OLLAMA ERROR {e}")
    return latencies


def print_stats(label: str, latencies: list[float]):
    if not latencies:
        print(f"  {label}: no data")
        return
    print(f"\n  {label}:")
    print(f"    Count:     {len(latencies)}")
    print(f"    Mean:      {mean(latencies):.0f}ms")
    print(f"    Median:    {median(latencies):.0f}ms")
    if len(latencies) > 2:
        print(f"    StdDev:    {stdev(latencies):.0f}ms")
    print(f"    Min:       {min(latencies):.0f}ms")
    print(f"    Max:       {max(latencies):.0f}ms")


async def main():
    parser = argparse.ArgumentParser(description="Zilli vs Ollama benchmark")
    parser.add_argument("--zilli-url", default="http://127.0.0.1:8900")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:7b")
    parser.add_argument("--requests", type=int, default=len(PROMPTS))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        print(f"Benchmarking Zilli at {args.zilli_url} vs Ollama at {args.ollama_url}")
        print(f"Model: {args.model}, Requests: {args.requests}, Concurrency: {args.concurrency}")
        print()

        print("--- Zilli (non-streaming) ---")
        zilli_lat = await benchmark_zilli(client, args.zilli_url, args.model, sem)
        RESULTS["zilli"] = zilli_lat

        print("\n--- Zilli (streaming, TTFT) ---")
        stream_ttft = await benchmark_zilli_stream(client, args.zilli_url, args.model, sem)
        RESULTS["zilli_stream"] = stream_ttft

        print("\n--- Ollama Direct ---")
        ollama_lat = await benchmark_ollama_direct(client, args.ollama_url, args.model, sem)
        RESULTS["ollama"] = ollama_lat

    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print_stats("Zilli (end-to-end)", RESULTS["zilli"])
    print_stats("Zilli Streaming (TTFT)", RESULTS["zilli_stream"])
    print_stats("Ollama Direct", RESULTS["ollama"])

    if RESULTS["zilli"] and RESULTS["ollama"]:
        z_mean = mean(RESULTS["zilli"])
        o_mean = mean(RESULTS["ollama"])
        ratio = z_mean / o_mean if o_mean else float("inf")
        print(f"\n  Zilli/Ollama latency ratio: {ratio:.2f}x")
        if ratio > 1.1:
            print(f"  Zilli adds {(ratio - 1) * 100:.0f}% overhead vs direct Ollama")
        elif ratio < 0.9:
            print(f"  Zilli is {(1 - ratio) * 100:.0f}% faster than direct Ollama (caching!)")
        else:
            print("  Zilli overhead is negligible")


if __name__ == "__main__":
    asyncio.run(main())
