from src.router_architecture import SemanticLLMRouter

router=SemanticLLMRouter()
tests = [
    ("مرحبا", "greeting"),
    ("كيف حالك؟", "greeting"),
    ("اشرح الحوسبة الكمومية", "complex"),
    ("اكتب كود بايثون", "complex"),
    ("ما هو الطقس اليوم؟", "unknown"),
]
for text, expected  in tests:
    route=router.predict_route(text)
    print(f"{text}->{route}(متوقع: {expected})")