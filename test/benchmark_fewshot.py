from src.router_architecture import SemanticLLMRouter

def run_benchmark():
    print("🚀 Starting Few-Shot Routing Benchmark...")
    router = SemanticLLMRouter()

    test_cases = [
        # Greetings
        ("hello", "Local Simulator"),
        ("how are you?", "Local Simulator"),
        ("مرحبا", "Local Simulator"),
        ("كيف حالك؟", "Local Simulator"),
        
        # Complex
        ("explain quantum computing", "gemini-2.5-pro"),
        ("write a python script", "gemini-2.5-pro"),
        ("اشرح الحوسبة الكمومية", "gemini-2.5-pro"),
        ("اكتب كود بايثون", "gemini-2.5-pro"),
        
        # Unknown/Simple
        ("what is the weather?", "gemini-2.5-flash"),
        ("who won the match?", "gemini-2.5-flash"),
    ]

    correct = 0
    total = len(test_cases)
    for query, expected in test_cases:
        route= router.predict_route(query)
        is_correct=(route==expected)
        if is_correct:
            correct+=1
        status= "✅" if is_correct else "❌"
        print(f"{status} Query: '{query}' -> Got: {route} | Expected: {expected}")
    accuracy=(correct/total)*100
    print(f"\n📊 Final Accuracy: {correct}/{total} ({accuracy:.2f}%)")
if __name__ == "__main__":
    run_benchmark()
