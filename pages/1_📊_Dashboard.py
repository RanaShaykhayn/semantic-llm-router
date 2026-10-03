import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path(__file__).parent.parent / "router_data.db"

USD_TO_SAR = 3.75

LABELS = {
    "ar": {
        "title": "📊 لوحة نتائج الراوتر", "total_queries": "إجمالي الاستعلامات",
        "cache_hit_rate": "نسبة إصابة الكاش", "total_cost": "إجمالي التكلفة الفعلية",
        "route_distribution": "توزيع المسارات", "avg_cost_per_route": "متوسط التكلفة لكل مسار",
        "full_log": "السجل الكامل", "no_data": "لا توجد بيانات بعد",
    },
    "en": {
        "title": "📊 Router Results Dashboard", "total_queries": "Total Queries",
        "cache_hit_rate": "Cache Hit Rate", "total_cost": "Total Actual Cost",
        "route_distribution": "Route Distribution", "avg_cost_per_route": "Avg Cost per Route",
        "full_log": "Full Log", "no_data": "No data yet",
    },
}


lang = st.sidebar.selectbox("Language / اللغة", ["ar", "en"])
currency = st.sidebar.selectbox("Currency / العملة", ["USD", "SAR"])
t = LABELS[lang]

st.title(t["title"])
# Fetch database records into a Pandas DataFrame for analysis
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM query_logs", conn)
conn.close()

if df.empty:
    st.warning(t["no_data"])
else:
    
    if currency == "SAR":
        df["display_cost"] = df["real_cost"] * USD_TO_SAR
        symbol = "﷼"
    else:
        df["display_cost"] = df["real_cost"]
        symbol = "$"

    col1, col2, col3 = st.columns(3)
    col1.metric(t["total_queries"], len(df))
    col2.metric(t["cache_hit_rate"], f"{(df['was_cache_hit'].sum() / len(df) * 100):.1f}%")
    col3.metric(t["total_cost"], f"{symbol}{df['display_cost'].sum():.5f}")

    st.subheader(t["route_distribution"])
    st.bar_chart(df['predicted_route'].value_counts())

    st.subheader(t["avg_cost_per_route"])
    cost_by_route = df.groupby('predicted_route')['display_cost'].mean().dropna()
    st.bar_chart(cost_by_route)

    st.subheader(t["full_log"])
    st.dataframe(df)