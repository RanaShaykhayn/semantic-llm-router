import os
from google import genai
import numpy as np
from dotenv import load_dotenv 
from google.genai import errors as genai_errors
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .database import log_query, get_estimated_cost
from .config import (
    REFERENCE_GREETINGS, REFERENCE_COMPLEX,
    CACHE_TTL_SECONDS, CACHE_MAX_SIZE,
    SIMILARITY_THRESHOLD_COMPLEX, SIMILARITY_THRESHOLD_GREETING, SIMILARITY_THRESHOLD_CACHE,
    GEMINI_FLASH_INPUT_PRICE, GEMINI_FLASH_OUTPUT_PRICE,
    GEMINI_PRO_INPUT_PRICE, GEMINI_PRO_OUTPUT_PRICE,
)

load_dotenv() 

class SemanticLLMRouter:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.gemini_pro = "gemini-2.5-pro"
        self.gemini_flash = "gemini-2.5-flash"
        self.semantic_cache = {} # reduce costs and improve performance
        self.CACHE_MAX_SIZE = CACHE_MAX_SIZE
        self.CACHE_TTL = CACHE_TTL_SECONDS
        self.reference_complex = REFERENCE_COMPLEX
        self._complex_vectors = np.array([self.get_embedding(c) for c in self.reference_complex])

        self.reference_greetings = REFERENCE_GREETINGS
        self._greeting_vectors = np.array([self.get_embedding(g) for g in self.reference_greetings])

    def preprocess_text(self, text: str | None) -> str:

        """Clean and normalize user text """
        
        if text is None or not text.strip():
            return ""
        cleaned = text.strip().lower()
        punctuations = ['?', '!', '.', ',', '،', '؟']
        for p in punctuations:
            cleaned = cleaned.replace(p, "")
        return " ".join(cleaned.split())
    
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(Exception)
    )

    def get_embedding(self, text: str) -> np.ndarray:

        """Generate vector embedding for the given text API"""

        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return np.array(response.embeddings[0].values)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute score"""
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm_product)

    def batch_cosine_similarity(self, vectors: np.ndarray, query: np.ndarray) -> np.ndarray:

        """Compute scores vector and a matrix of vectors in parallel"""

        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(vectors, axis=1)
        denom = vector_norms * query_norm
        denom = np.where(denom == 0, 1e-10, denom) 
        return np.dot(vectors, query) / denom

    def check_semantic_cache(self, user_vector:  np.ndarray) -> tuple [str | None, float | None]:

        """check a similar cache and return response and cost or none"""

        self.evict_expired_cache() 
        if not self.semantic_cache:
            return None, None
        cache_keys = list(self.semantic_cache.keys())
        cache_vectors = np.array([self.semantic_cache[k]["vector"] for k in cache_keys])
        # Compare the current vector against all stored queries
        similarities = self.batch_cosine_similarity(cache_vectors, user_vector)
        max_idx = np.argmax(similarities)
        if similarities[max_idx] >= SIMILARITY_THRESHOLD_CACHE:
            matched_query = cache_keys[max_idx]
            self.semantic_cache[matched_query]["last_used"] = time.time()
            return self.semantic_cache[matched_query]["response"], self.semantic_cache[matched_query]["cost"]
        return None, None

    def update_cache(self, query: str, query_vector: np.ndarray, response: str, real_cost: float) -> None:

        """store in new query andleast recently used item if full"""

        if len(self.semantic_cache) >= self.CACHE_MAX_SIZE:
            least_used = min(self.semantic_cache, key=lambda k: self.semantic_cache[k]["last_used"])
            del self.semantic_cache[least_used]
        self.semantic_cache[query] = {
            "vector": query_vector,
            "response": response,
            "cost": real_cost,
            "timestamp": time.time(),
            "last_used": time.time()}
        

    def evict_expired_cache(self) -> None:
        """remove expired entries and based on their time-to-live"""
        current_time = time.time()
        expired = [q for q, d in self.semantic_cache.items() if (current_time - d["timestamp"]) > self.CACHE_TTL]
        for q in expired:
            del self.semantic_cache[q]

    def prepare_routing_engine(self, user_vector: np.ndarray) -> str:

        """determine the target route and greeting prompts"""

        complex_similarities = self.batch_cosine_similarity(self._complex_vectors, user_vector)
        # Extract the highest similarity score (best match)
        max_complex_similarity = float(np.max(complex_similarities))
        print(f"max_complex_similarity: {max_complex_similarity:.4f}")
        greet_similarities = self.batch_cosine_similarity(self._greeting_vectors, user_vector)
        max_greet_similarity = float(np.max(greet_similarities))

        if max_complex_similarity > max_greet_similarity and max_complex_similarity >= SIMILARITY_THRESHOLD_COMPLEX:
            return self.gemini_pro
        
        elif max_greet_similarity >= SIMILARITY_THRESHOLD_GREETING:
            return "Local Simulator"
        else:
            return self.gemini_flash # Fallback route: neither complex nor greeting-like

    def predict_route(self, user_input: str | None) -> str:
        """input string without executing model generation or cache updates"""
        cleaned_text = self.preprocess_text(user_input)
        if not cleaned_text:
            return "Empty Input"
        user_vector = self.get_embedding(cleaned_text)
        return self.prepare_routing_engine(user_vector)

    def execute_routing(self, user_input: str | None) -> dict:

        """Process a query full pipeline including cache lookup, routing, execution, and telemetry logging"""
        
        start_time = time.time()
        
        cleaned_text = self.preprocess_text(user_input)
        if not cleaned_text:
            return {"route": "Empty Input", "response": "Please enter a valid text.", "money_saved": 0.0}
        user_vector = self.get_embedding(cleaned_text)
        cached_response, cached_cost = self.check_semantic_cache(user_vector)
        if cached_response:
            latency = (time.time() - start_time) * 1000
            log_query({
                "raw_query": user_input, "cleaned_query": cleaned_text,
                "predicted_route": "Semantic Cache (Hit)", "was_cache_hit": True,
                "latency_ms": latency,
            })
            return {"route": "Semantic Cache (Hit)", "response": cached_response, "money_saved": cached_cost}
        target_route = self.prepare_routing_engine(user_vector)

        try:
            if target_route == "Local Simulator":
                response = "Hello! How can I assist you today?"
                input_tokens = output_tokens = real_cost = 0
            else: 
                api_response = self.client.models.generate_content(model=target_route, contents=user_input)
                response = api_response.text
                usage = api_response.usage_metadata
                input_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
                if target_route == self.gemini_flash:
                    input_price, output_price = GEMINI_FLASH_INPUT_PRICE, GEMINI_FLASH_OUTPUT_PRICE
                else:
                    input_price, output_price = GEMINI_PRO_INPUT_PRICE, GEMINI_PRO_OUTPUT_PRICE

                real_cost = (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)

            self.update_cache(cleaned_text, user_vector, response, real_cost)
            latency = (time.time() - start_time) * 1000

            log_query({
                "raw_query": user_input, "cleaned_query": cleaned_text,
                "predicted_route": target_route, "was_cache_hit": False,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "real_cost": real_cost, "latency_ms": latency,
            })

            return {"route": target_route, "response": response, "money_saved": 0.0}

        except genai_errors.ClientError as e:
            latency = (time.time() - start_time) * 1000
            estimated_tokens = len(user_input.split())

            if e.code == 429:
                friendly_message = (
                    f"⚠️ Rate limit reached for model{target_route}  "
                    f"   Approximate word count: {estimated_tokens}"
                )
            else:
                friendly_message = f"⚠️ Please try again later {e.code})"

            log_query({
                "raw_query": user_input, "cleaned_query": cleaned_text,
                "predicted_route": "Fallback", "was_cache_hit": False, "latency_ms": latency,
            })
            return {"route": "Fallback", "response": friendly_message, "money_saved": 0.0}

        except Exception:
            latency = (time.time() - start_time) * 1000
            log_query({
                "raw_query": user_input, "cleaned_query": cleaned_text,
                "predicted_route": "Fallback", "was_cache_hit": False, "latency_ms": latency,
            })
            return {"route": "Fallback", "response": "⚠️ An unexpected error occurred in the system", "money_saved": 0.0}
   
 


 


