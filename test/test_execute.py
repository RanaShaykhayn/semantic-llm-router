from src.router_architecture import SemanticLLMRouter

router = SemanticLLMRouter()
result = router.execute_routing("What is 2+2?")

print("=" * 60)
print(f"Route: {result['route']}")
print(f"Response: {result['response'][:100]}...")
print("=" * 60)