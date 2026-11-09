import time
import numpy as np
from src.database import get_estimated_cost
from src.router_architecture import SemanticLLMRouter

def run_full_benchmark():
    router = SemanticLLMRouter()
    start_time = time.time()
    test_queries = [
        "Hello, how are you?",
        "How do I check my order status?",
        "Explain quantum computing in simple terms",
        "Code a python script for web scraping",
        "Summarize the history of artificial intelligence in 3 sentences.",
        "How does a database index work?",
        "Write a quick sort algorithm in Python.",
        "What are the main causes of inflation?",
        "Hi there" ]
    test_queries = test_queries* 5  
    total_samples = len(test_queries)

    print("===running full benchmark===")
    latencies = []
    for item in test_queries :
        req_start = time.time()
        result= router.execute_routing(item)
        req_duration = time.time() - req_start
        latencies.append(req_duration)
        print(f"Route: {result['route']} | Response: {result['response']}")
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    print(f"\n=== Benchmark Summary ({total_samples} samples) ===")
    print(f"p95 Latency: {p95_latency:.4f}s (95% of requests were faster than this)")
    print(f"p99 Latency: {p99_latency:.4f}s (Tail latency / worst cases)")
    print(f"\n=== Benchmark Summary ({total_samples} samples) ===")
    for route in ["gemini-2.5-flash", "gemini-2.5-pro", "Local Simulator"]:
        estimated_cost, sample_size = get_estimated_cost(route)
        if estimated_cost is not None:
            print(f"{route}: estimated avg cost ${estimated_cost:.6f} (from {sample_size} past queries)")
        else:
            print(f"{route}: not enough historical data yet ({sample_size} samples)")

    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f}s")
if __name__=="__main__":
    run_full_benchmark() 
   