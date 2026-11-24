import os
from google import genai
from src.router_architecture import SemanticLLMRouter
from dotenv import load_dotenv

load_dotenv()

class ReActAgent:
    def __init__(self):
        self.router = SemanticLLMRouter()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def run(self, user_input: str) -> str:
        max_iterations = 3
        history = ""
        
        for i in range(max_iterations):
            prompt = f"""
You are an AI assistant. You have a tool called 'router' that can answer questions about company policy.
User question: {user_input}
History of your actions: {history}

What should you do? Answer ONLY with:
- 'USE_ROUTER' if you need to use the router to find information.
- 'FINAL_ANSWER' if you already have the answer or cannot find it.
"""
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            decision = response.text.strip().upper()
            
            if "USE_ROUTER" in decision:
                print(f"🤔 Iteration {i+1}: Agent decided to use the Router.")
                result = self.router.execute_routing(user_input)
                history += f"\n- Used Router. Result: {result['response']}"
                print(f"👀 Observed: {result['response'][:100]}...")
                
            elif "FINAL_ANSWER" in decision:
                print(f"✅ Iteration {i+1}: Agent decided to stop.")
                final_prompt = f"Based on this history, answer the user: {history}\nUser: {user_input}"
                final_response = self.client.models.generate_content(model="gemini-2.5-flash", contents=final_prompt)
                return final_response.text

        return f"⚠️ Agent reached max iterations ({max_iterations}) without a final answer."