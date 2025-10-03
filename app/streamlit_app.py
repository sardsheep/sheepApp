


import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient

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

# 3) Minimal query (adjust measurement and fields to match your ingest)
hours = st.slider("Time window (hours)", 1, 72, 24)
sheep_id = st.text_input("Filter by sheep_id (optional)")
label = st.selectbox("Filter by label (optional)", ["", "grazing", "ruminating", "walking", "other"])

filters = []
if sheep_id:
    filters.append(f'  |> filter(fn: (r) => r["sheep_id"] == "{sheep_id}")')
if label:
    filters.append(f'  |> filter(fn: (r) => r["label"] == "{label}")')
flt_block = "\n".join(filters)

flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: -{hours}h)
  |> filter(fn: (r) => r._measurement == "sheep_behavior_pred")
  |> filter(fn: (r) => r._field == "confidence")
{flt_block}
  |> limit(n: 5)
'''

st.code(flux, language="flux")

# 4) Run query & show results
try:
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        tables = client.query_api().query_data_frame(flux)
    df = pd.concat(tables, ignore_index=True) if isinstance(tables, list) else tables
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Query returned no rows. Try a longer time window, remove filters, or check that data exists in the bucket.")
    else:
        # Standardize column names
        if "_time" in df.columns: df = df.rename(columns={"_time":"time"})
        if "_value" in df.columns: df = df.rename(columns={"_value":"confidence"})
        keep_cols = [c for c in ["time", "confidence", "label", "sheep_id", "_measurement", "_field"] if c in df.columns]
        st.dataframe(df[keep_cols].sort_values("time").head(200))
        if "time" in df.columns and "confidence" in df.columns:
            st.line_chart(df.set_index("time")["confidence"])
except Exception as e:
    st.error("Query failed. This is usually a token scope, org/bucket name, or region URL issue.")
    st.exception(e)
