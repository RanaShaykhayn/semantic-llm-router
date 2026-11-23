import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class RAGEvaluator:
    def __init__(self,retriver):
        self.retriver=retriver
        self.client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def is_correct(self, query:str, ret_text:str, ground_truth:str)->bool:
        prompt=f"""You are a strict evaluator.
        Question: {query}
Retrieved Context: {ret_text}
Expected Answer: {ground_truth}

Does the Retrieved Context contain the information needed to answer the question correctly?
Answer ONLY with 'YES' or 'NO'."""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return "YES" in response.text.upper()

    def evaluate(self,golden_dataset:list)->dict:
        correct=0
        total=len(golden_dataset)
        for i in golden_dataset:
            query=i["query"]
            ground_truth=i["ground_truth"]
            results=self.retriver.search(query, top_k=1)
            if not results:
                continue
            ret_text=results[0]["text"]

            if self.is_correct(query, ret_text, ground_truth):
                correct+=1
                print(f"✅ Query: '{query}' -> Correct.")
            else:
                print(f"❌ Query: '{query}' -> Incorrect. (Retrieved: '{ret_text}')")

        precision = correct / total if total > 0 else 0.0
        print(f"\n📊 Precision: {precision:.2%} ({correct}/{total})")
        
        return {"precision": precision, "correct": correct, "total": total}   