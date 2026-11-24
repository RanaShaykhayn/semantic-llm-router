from src.router_architecture import SemanticLLMRouter
from src.agent import ReActAgent

class HybridRouter:
    def __init__(self):
        self.router=SemanticLLMRouter()
        self.agent=ReActAgent()

    def route(self, user_input:str)->dict:

        if any (word in user_input.lower() for word in ["leave", "policy", "work", "insurance", "bonus"]):
            print("🧠 Question is complex. Routing to ReAct Agent...")
            response = self.agent.run(user_input)
            return {"route": "Agentic (ReAct)", "response": response}
        else:
            print("⚡ Question is simple. Routing to Deterministic Router...")
            return self.router.execute_routing(user_input)
