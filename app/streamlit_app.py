import streamlit as st
import pandas as pd
import pyarrow as pa
from influxdb_client import InfluxDBClient            # v2 client (health)
from influxdb_client_3 import InfluxDBClient3         # v3 client (SQL)

st.set_page_config(page_title="Sheep Behavior — SQL", layout="wide")
st.title("🐑 Sheep Behavior — InfluxDB Cloud (SQL)")

# --- 1) Secrets ---
try:
    cfg = st.secrets["influx"]
    URL   = cfg["url"]     # e.g., "https://eu-central-1-1.aws.cloud2.influxdata.com"
    TOKEN = cfg["token"]   # READ token
    ORG   = cfg["org"]     # org name or ID
    DB    = cfg["bucket"]  # <-- SQL "database" == your bucket
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

# --- 2) Connectivity check (v2 health) ---
try:
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        health = client.health()
        st.success(f"Influx health: {health.status} — {health.message}")
except Exception as e:
    st.error("Could not reach InfluxDB. Check URL (region!), org, and token scope.")
    st.exception(e)
    st.stop()



# RADIO: 1 = Ram, 2 = EWE (+ 'Any' to disable the filter)
choice = st.radio(
    "Sheep type",
    options=["Any", "Ram ", "Ewe "],
    index=0
)

# Normalize the selection to the label you'd store in DB
# If your column stores text like 'Ram'/'EWE', keep this:
type_value = None
if choice == "Ram":
    type_value = "Ram"
elif choice == "Ewe":
    type_value = "Ewe"

# Build the WHERE clause part
# Text column version (default). If your column is named differently, change 'sheep_type'.
type_clause = "" if type_value is None else f"  AND LOWER(sheep_type) = LOWER('{type_value}')\n"

# --- If your DB stores numbers (1/2) instead of text, use this instead of the line above:
# num_value = None
# if choice == "Ram (1)":
#     num_value = 1
# elif choice == "EWE (2)":
#     num_value = 2
# type_clause = "" if num_value is None else f"  AND sheep_type = {num_value}\n"



# --- 3) Inputs + SQL ---
sheep_id = st.text_input("Sheep ID", value="1")
days = st.slider("Look-back window (days)", min_value=1, max_value=365, value=364, help="Free plan retains ~30 days")

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

# --- 4) Run SQL (v3 client) and normalize to pandas ---
try:
    with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        result = client.query(sql)  # may return PyArrow Table or pandas DataFrame depending on client version

    # Normalize to pandas.DataFrame
    if isinstance(result, pd.DataFrame):
        df = result
    elif isinstance(result, pa.Table):
        df = result.to_pandas()
    elif isinstance(result, list) and result and isinstance(result[0], pa.Table):
        df = pd.concat([t.to_pandas() for t in result], ignore_index=True)
    else:
        # Fallback: try to build a DataFrame
        df = pd.DataFrame(result)

    if df is None or df.empty:
        st.info("No rows returned. Try another Sheep ID, increase the window, or confirm recent data exists (Free plan ≈ 30 days).")
    else:
        st.dataframe(df)
        if {"time", "confidence"}.issubset(df.columns):
            # Ensure time is datetime index for the chart
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            st.line_chart(df.set_index("time")["confidence"])
except Exception as e:
    st.error("SQL query failed. Check URL/Org/Token/Database (bucket) and retention.")
    st.exception(e)
