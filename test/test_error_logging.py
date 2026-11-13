from src.router_architecture import SemanticLLMRouter
import sqlite3

def test_error_logging():
    print("🚀 Testing Error Logging...")
    router = SemanticLLMRouter()

    print("\n--- Testing Safe Query ---")
    result = router.execute_routing("What is 2+2?")
    print(f"Route: {result['route']}")

    conn = sqlite3.connect('router_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT predicted_route, error_message FROM query_logs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    print(f"\n📊 Last DB Entry:")
    print(f"Route: {row[0]}")
    print(f"Error Message: {row[1]}")
    
    assert row[0] == "gemini-2.5-flash"
    assert row[1] is None 
    print("\n✅ Error logging works correctly!")

if __name__ == "__main__":
    test_error_logging()