from src.router_architecture import SemanticLLMRouter

def test_rag():
    print("🚀 Testing RAG Integration...")
    router = SemanticLLMRouter()

    print("\n--- Testing RAG Query ---")
    result = router.execute_routing("How many days of annual leave do I have?")
    print(f"Route: {result['route']}")
    print(f"Response: {result['response'][:200]}...")

    print("\n--- Testing General Query ---")
    result2 = router.execute_routing("What is 2+2?")
    print(f"Route: {result2['route']}")
    print(f"Response: {result2['response'][:100]}...")
    
    print("\n✅ RAG Integration test complete!")

if __name__ == "__main__":
    test_rag()