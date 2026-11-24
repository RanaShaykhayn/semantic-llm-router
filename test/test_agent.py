from src.agent import ReActAgent

def test_agent():
    print("🚀 Testing ReAct Agent...")
    agent = ReActAgent()
    result = agent.run("How many days of annual leave do I have?")
    print(f"\n🤖 Final Agent Response: {result}")

if __name__ == "__main__":
    test_agent()