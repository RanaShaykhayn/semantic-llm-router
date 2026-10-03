# test/test_router.py
from src.router_architecture import SemanticLLMRouter

def test_simple_routing():
    router = SemanticLLMRouter()
    result = router.execute_routing("مرحباً")
    print(f"النتيجة: {result['route']}")
    # Assert that the routing process has successfully identified a path
    assert result['route'] is not None
# Execute tests only if the script is run directly
if __name__ == "__main__":
    test_simple_routing()
    print("الاختبار مر بنجاح! ✅")