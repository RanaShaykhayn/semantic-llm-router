from src.router_architecture import SemanticLLMRouter

def test_pydantic_routing():
    print("🚀 Starting Pydantic test...")
    router = SemanticLLMRouter()

    result = router.execute_routing("What is 2+2?")
    
    print(f"Route: {result['route']}")
    print(f"Response: {result['response'][:50]}...")
    print(f"Latency: {result['latency_ms']:.2f} ms")

    assert isinstance(result, dict), "Result must be a dict"
    assert "route" in result, "Result must contain 'route'"
    assert "latency_ms" in result, "Result must contain 'latency_ms'"
    print("✅ Pydantic test passed!")

if __name__ == "__main__":
    test_pydantic_routing()