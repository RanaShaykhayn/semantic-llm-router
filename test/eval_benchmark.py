import time
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
        "Hi there" ]
    
    total_samples = len(test_queries)

    print("===running full benchmark===")
    
    for item in test_queries :
        result= router.execute_routing(item)
        print(f"Route: {result['route']} | Response: {result['response']}")
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