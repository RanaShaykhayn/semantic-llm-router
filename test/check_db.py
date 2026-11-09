import sqlite3

conn = sqlite3.connect('router_data.db')
cursor = conn.cursor()
cursor.execute('SELECT preprocess_ms, embedding_ms, cache_ms, routing_ms, generation_ms, latency_ms FROM query_logs ORDER BY id DESC LIMIT 1')
row = cursor.fetchone()

if row:
    print(f"preprocess: {row[0]:.2f} ms")
    print(f"embedding:  {row[1]:.2f} ms")
    print(f"cache:      {row[2]:.2f} ms")
    print(f"routing:    {row[3]:.2f} ms")
    print(f"generation: {row[4]:.2f} ms")
    print(f"latency:    {row[5]:.2f} ms")
else:
    print("No rows found")

conn.close()