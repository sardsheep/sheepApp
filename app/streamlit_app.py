import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient

st.set_page_config(page_title="Sheep Behavior", layout="wide")

# --- Config from secrets ---
cfg = st.secrets["influx"]   # Streamlit Cloud: set in Secrets UI; local: secrets.toml
URL, TOKEN, ORG, BUCKET = cfg["url"], cfg["token"], cfg["org"], cfg["bucket"]

# --- Query helper ---
@st.cache_data(ttl=60)
def query_confidence(hours=24, sheep_id=None, label=None):
    sheep_filter = f'  |> filter(fn: (r) => r["sheep_id"] == "{sheep_id}")\n' if sheep_id else ""
    label_filter = f'  |> filter(fn: (r) => r["label"] == "{label}")\n' if label else ""
    flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: -{hours}h)
  |> filter(fn: (r) => r._measurement == "sheep_behavior_pred")
  |> filter(fn: (r) => r._field == "confidence")
{sheep_filter}{label_filter}
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> yield(name: "mean_conf")
'''
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        dfs = client.query_api().query_data_frame(flux)
    if isinstance(dfs, list) and dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = dfs
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"_time": "time", "_value": "confidence"})
    keep = [c for c in ["time", "confidence", "label", "sheep_id"] if c in df.columns]
    return df[keep].sort_values("time")

st.title("🐑 Sheep Behavior — Predictions")

col1, col2, col3 = st.columns(3)
with col1: hours = st.slider("Time window (hours)", 1, 72, 24)
with col2: sheep = st.text_input("Sheep ID (optional)")
with col3: label = st.selectbox("Label (optional)", ["", "grazing", "ruminating", "walking", "other"])

df = query_confidence(hours, sheep_id=(sheep or None), label=(label or None))

if df.empty:
    st.info("No data returned for the selected filters/time window.")
else:
    st.line_chart(df.set_index("time")["confidence"])
    st.caption(f"{len(df)} points")
    st.dataframe(df.tail(200))

