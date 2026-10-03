CREATE TABLE IF NOT EXISTS query_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_query TEXT NOT NULL,
    cleaned_query TEXT NOT NULL,
    true_label INTEGER ,
    predicted_route TEXT NOT NULL,
    was_cache_hit BOOLEAN NOT NULL DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    real_cost REAL,
    latency_ms REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS evaluation_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    total_samples INTEGER NOT NULL,
    excluded_samples INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    tp INTEGER NOT NULL,
    fp INTEGER NOT NULL,
    fn INTEGER NOT NULL,
    tn INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);