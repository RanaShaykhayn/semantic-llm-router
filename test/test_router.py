import pytest
from src.router_architecture import SemanticLLMRouter

@pytest.fixture(scope="module")
def router():
    return SemanticLLMRouter()

class TestPreprocessText:
    def test_basic(self, router):
        assert router.preprocess_text("Hello, World!")=="hello world"

    def test_uppercase(self, router):
        assert router.preprocess_text("HELLO") == "hello"

    def test_spaces(self, router):
        assert router.preprocess_text("  hello   world  ") == "hello world"

    def test_none(self, router):
         assert router.preprocess_text(None) == ""
    
    def test_empty(self, router):
        assert router.preprocess_text("") == ""

    def test_arabic(self, router):
        assert router.preprocess_text("مرحبا؟") == "مرحبا"
