import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient            # v2 client (for health check)
from influxdb_client_3 import InfluxDBClient3         # v3 client (for SQL)

st.set_page_config(page_title="Sheep Behavior — SQL", layout="wide")
st.title("🐑 Sheep Behavior — InfluxDB Cloud (SQL)")

# --- 1) Read secrets ---
try:
    cfg = st.secrets["influx"]
    URL   = cfg["url"]     # e.g., "https://eu-central-1-1.aws.cloud2.influxdata.com"
    TOKEN = cfg["token"]   # READ token
    ORG   = cfg["org"]     # org name or ID
    DB    = cfg["bucket"]  # <- in SQL this is the DATABASE (your bucket)
except Exception as e:
    st.error("Missing or malformed secrets. Set [influx] url/token/org/bucket in Streamlit Secrets.")
    st.exception(e)
    st.stop()

with st.expander("Connection config (sanitized)"):
    st.write({
        "url": URL,
        "org": ORG,
        "database/bucket": DB,
        "token_prefix": TOKEN[:6] + "..." if isinstance(TOKEN, str) and len(TOKEN) > 6 else "short/invalid",
    })

# --- 2) Connectivity check (optional) using v2 health endpoint ---
try:
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        health = client.health()
        st.success(f"Influx health: {health.status} — {health.message}")
except Exception as e:
    st.error("Could not reach InfluxDB. Check URL (region!), org, and token scope.")
    st.exception(e)
    st.stop()

# --- 3) Inputs + SQL (your query) ---
sheep_id = st.text_input("Sheep ID", value="1")
days = st.slider("Look-back window (days)", min_value=1, max_value=365, value=30, help="Free plan retains ~30 days")

sql = f"""
SELECT time, confidence, label, sheep_id
FROM sheep_behavior_pred
WHERE sheep_id = '{sheep_id}'
  AND time >= now() - INTERVAL '{days} days'
ORDER BY time DESC
LIMIT 1000;
"""

st.subheader("SQL")
st.code(sql, language="sql")

# --- 4) Run SQL via v3 client ---
try:
    with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        df: pd.DataFrame = client.query(sql)

    if df is None or df.empty:
        st.info("No rows returned. Try another Sheep ID, increase the window, or confirm recent data exists (Free plan ≈ 30 days).")
    else:
        st.dataframe(df)
        if {"time", "confidence"}.issubset(df.columns):
            st.line_chart(df.set_index("time")["confidence"])
except Exception as e:
    st.error("SQL query failed. Check URL/Org/Token/Database (bucket) and retention.")
    st.exception(e)
