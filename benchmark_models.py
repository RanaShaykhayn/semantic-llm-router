import time
import numpy as np 
import os 
from dotenv import load_dotenv
from google import genai
from src.config import(
    GEMINI_FLASH_INPUT_PRICE, GEMINI_FLASH_OUTPUT_PRICE,
    GEMINI_PRO_INPUT_PRICE, GEMINI_PRO_OUTPUT_PRICE,
)
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL   = "gemini-pro-latest"


def measure_one_call(client, model_name, query):
    start= time.time()
    try:
        response= client.models.generate_content(
            model=model_name,
            contents=query
        )
        latency_ms=(time.time()-start)*1000
        usage=response.usage_metadata
        input_tokens=usage.prompt_token_count
        output_tokens=usage.candidates_token_count
        if model_name==FLASH_MODEL:
            in_price, out_price=GEMINI_FLASH_INPUT_PRICE,GEMINI_FLASH_OUTPUT_PRICE
        else:
            in_price, out_price=GEMINI_PRO_INPUT_PRICE,GEMINI_PRO_OUTPUT_PRICE

        cost=(input_tokens/1_000_000 *in_price) + \
             (output_tokens/1_000_000*out_price)
        return {
                "success": True,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "error_type": None,
                "response_text": response.text,
            }
    except Exception as e:
            print(f"      [DEBUG] {type(e).__name__}: {e}")
            latency_ms = (time.time() - start) * 1000
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in error_msg:
              error_type = "rate_limit"
            elif "timeout" in error_msg or "deadline" in error_msg:
              error_type = "timeout"
            else:
              error_type = "other"
            return {
            "success": False,
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "error_type": error_type,
            "response_text": None,
        }
QUERIES = [
    "What is 2+2?",
    "Explain what a Python list is in one sentence.",
    "Write a Python function that adds two numbers.",
]


def run_benchmark():
    results = {FLASH_MODEL: []}

    for model in [FLASH_MODEL]:
        print(f"\n=== Testing {model} ===")
        for i, query in enumerate(QUERIES, start=1):
            print(f"  [{i}/{len(QUERIES)}] {query[:50]}...")
            r = measure_one_call(client, model, query)
            results[model].append(r)
            status = "OK" if r["success"] else f"FAIL ({r['error_type']})"
            print(f"      {status} | {r['latency_ms']:.0f}ms | ${r['cost']:.6f}")

    return results
if __name__ == "__main__":
    results = run_benchmark()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for model in [FLASH_MODEL]:
        model_results = results[model]
        successful = [r for r in model_results if r["success"]]
        failed = [r for r in model_results if not r["success"]]

        if not successful:
            print(f"\n{model}: All {len(model_results)} calls failed")
            continue

        latencies = [r["latency_ms"] for r in successful]
        costs = [r["cost"] for r in successful]
        in_tokens = sum(r["input_tokens"] for r in successful)
        out_tokens = sum(r["output_tokens"] for r in successful)

        print(f"\n--- {model} ---")
        print(f"  Calls: {len(successful)} OK / {len(failed)} failed")
        print(f"  Latency p50: {np.percentile(latencies, 50):.0f} ms")
        print(f"  Latency p95: {np.percentile(latencies, 95):.0f} ms")
        print(f"  Latency p99: {np.percentile(latencies, 99):.0f} ms")
        print(f"  Total tokens: {in_tokens} in / {out_tokens} out")
        print(f"  Total cost: ${sum(costs):.6f}")