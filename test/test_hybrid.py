from src.hybrid_router import HybridRouter

def test_hybrid():
    print("🚀 Testing Hybrid Router...")
    hybrid=HybridRouter()
    print("\n--- Simple Query ---")
    result1 = hybrid.route("What is 2+2?")
    print(f"Route: {result1['route']}")
    print("\n--- Complex Query ---")
    result2 = hybrid.route("How many days of annual leave do I have?")
    print(f"Route: {result2['route']}")

if __name__ == "__main__":
    test_hybrid()