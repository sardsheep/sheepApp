


import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient
#from influxdb_client_3 import InfluxDBClient3  # SQL client

st.set_page_config(page_title="Sheep Behavior (diagnostic)", layout="wide")
st.title("🐑 Sheep Behavior — InfluxDB Cloud diagnostic")

# 1) Read secrets (Streamlit Cloud: set in Secrets UI; local: .streamlit/secrets.toml)
try:
    cfg = st.secrets["influx"]
    URL   = cfg["url"]     # e.g., "https://eu-central-1-1.aws.cloud2.influxdata.com"
    TOKEN = cfg["token"]   # READ token
    ORG   = cfg["org"]     # org name or ID
    BUCKET= cfg["bucket"]  # e.g., "sheep_pred"
except Exception as e:
    st.error("Missing or malformed secrets. Set [influx] url/token/org/bucket in Streamlit Secrets.")
    st.exception(e)
    st.stop()

# Show sanitized config to confirm values (token hidden)
with st.expander("Connection config (sanitized)"):
    st.write({
        "url": URL,
        "org": ORG,
        "bucket": BUCKET,
        "token_prefix": TOKEN[:6] + "..." if isinstance(TOKEN, str) and len(TOKEN) > 6 else "short/invalid",
    })

# 2) Connectivity check
try:
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        health = client.health()
        st.success(f"Influx health: {health.status} — {health.message}")
except Exception as e:
    st.error("Could not reach InfluxDB. Check URL (region!), org, and token scope.")
    st.exception(e)
    st.stop()

# secrets
cfg = st.secrets["influx"]
URL, TOKEN, ORG, DB = cfg["url"], cfg["token"], cfg["org"], cfg["bucket"]

sheep_id = st.text_input("Sheep ID", value="1")
days = st.slider("Look-back window (days)", 1, 365, 30, help="Free plan retains ~30 days")

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

try:
    with InfluxDBClient(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        df: pd.DataFrame = client.query(sql)
    if df.empty:
        st.info("No rows returned. Try a smaller Sheep ID, increase the window, or confirm recent data exists (remember Free plan ≈ 30 days).")
    else:
        st.dataframe(df)
        if {"time","confidence"}.issubset(df.columns):
            st.line_chart(df.set_index("time")["confidence"])
except Exception as e:
    st.error("SQL query failed. Check URL/Org/Token/Database (bucket) and that your plan retention covers the requested window.")
    st.exception(e)

