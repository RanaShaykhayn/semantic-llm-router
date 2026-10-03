import sqlite3
from pathlib import Path

db_path=Path(__file__).parent.parent/"router_data.db"
schema_path = Path(__file__).parent.parent / "data" / "schema.sql"
# Initialize database: run once to create schema/tables
def init_db():
    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()
    # Load database schema from an external configuration file
    with open(schema_path, mode='r', encoding="utf-8")as f:
        schema=f.read()

    cursor.executescript(schema)
    conn.commit()
    conn.close()
    print("successful ac")
# Log request metadata: document every system interaction
def log_query(data:dict) ->None:
    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()
    cursor.execute("""
INSERT INTO query_logs( raw_query, cleaned_query, true_label, predicted_route,
was_cache_hit, input_tokens, output_tokens, real_cost, latency_ms)

VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) 
                   """, (
            data.get("raw_query"),
            data.get("cleaned_query"),
            data.get("true_label"),       
            data.get("predicted_route"),
            data.get("was_cache_hit", False),
            data.get("input_tokens"),     
            data.get("output_tokens"),    
            data.get("real_cost"),       
            data.get("latency_ms"),
        ),
 )
    
    conn.commit()
    conn.close()

def log_evaluation_run(data: dict) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluation_runs (
            total_samples,
            excluded_samples,
            accuracy,
            precision_score,
            recall,
            f1,
            tp,
            fp,
            fn,
            tn
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("total_samples"),
        data.get("excluded_samples"),
        data.get("accuracy"),
        data.get("precision_score"),
        data.get("recall"),
        data.get("f1"),
        data.get("tp"),
        data.get("fp"),
        data.get("fn"),
        data.get("tn")
    ))
    conn.commit()
    conn.close()
# Smart estimator: predict costs based on historical usage patterns
def get_estimated_cost(route: str) -> tuple[float, int] | tuple[None, int]:
  
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT AVG(real_cost), COUNT(*)
        FROM query_logs
        WHERE predicted_route = ? AND real_cost IS NOT NULL
        """,
        (route,)
    )
    avg_cost, sample_size = cursor.fetchone()
    conn.close()

    if sample_size < 5:
        return None, sample_size
    return avg_cost, sample_size

if __name__=="__main__":
 init_db()
    

