import streamlit as st
import pandas as pd
import pyarrow as pa
from influxdb_client import InfluxDBClient            # v2 client (health)
from influxdb_client_3 import InfluxDBClient3         # v3 client (SQL)

from datetime import datetime, timedelta, timezone, time as dtime
try:
    from zoneinfo import ZoneInfo
    TZ_LOCAL = ZoneInfo("Europe/Rome")  # change if needed
except Exception:
    TZ_LOCAL = timezone.utc  # fallback

st.set_page_config(page_title="Sheep Behavior — SQL", layout="wide")
st.title("🐑 Sheep Behavior")

# --- 1) Secrets ---
try:
    cfg = st.secrets["influx"]
    URL   = cfg["url"]
    TOKEN = cfg["token"]
    ORG   = cfg["org"]
    DB    = cfg["bucket"]
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

# --- 3) Inputs: explicit start/end date & time (local) ---
today_local = datetime.now(TZ_LOCAL).date()
default_start_date = today_local - timedelta(days=30)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start date (local)", value=default_start_date)
    start_time = st.time_input("Start time (local)", value=dtime(0, 0))
with c2:
    end_date = st.date_input("End date (local)", value=today_local)
    end_time = st.time_input("End time (local)", value=dtime(23, 59))

start_dt_local = datetime.combine(start_date, start_time, tzinfo=TZ_LOCAL)
end_dt_local   = datetime.combine(end_date,   end_time,   tzinfo=TZ_LOCAL)

if start_dt_local > end_dt_local:
    st.error("Start datetime must be before end datetime.")
    st.stop()

start_iso = start_dt_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
end_iso   = end_dt_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

st.caption(
    f"Querying from **{start_dt_local.strftime('%Y-%m-%d %H:%M %Z')}** "
    f"to **{end_dt_local.strftime('%Y-%m-%d %H:%M %Z')}** "
    f"(UTC: {start_iso} → {end_iso})"
)

# --- 4) Sheep ID multiselect (1..10). If none selected -> include all 10 ---
ALL_SHEEP_IDS = [str(i) for i in range(1, 11)]
selected_ids = st.multiselect(
    "Sheep IDs (optional)",
    options=ALL_SHEEP_IDS,
    default=[],
    help="Pick one or more IDs (1–10). Leave empty to include all 10."
)

def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

if selected_ids:
    id_in_list = ",".join(_q(sid) for sid in selected_ids)
else:
    # default to all 10 if nothing selected
    id_in_list = ",".join(_q(sid) for sid in ALL_SHEEP_IDS)

sheep_clause = f"  AND sheep_id IN ({id_in_list})\n"

# --- 5) Behaviour multiselect (8 behaviours). If none selected -> show all ---
ALL_BEHAVIOURS = [
    "flehmen", "grazing", "head-butting", "lying",
    "mating", "running", "standing", "walking",
]
selected_behaviours = st.multiselect(
    "Predicted Behaviour (optional filter)",
    options=ALL_BEHAVIOURS,
    default=[],
    help="Pick one or more behaviours. Leave empty to include all."
)

behaviour_clause = ""
if selected_behaviours:
    beh_in_list = ",".join("'" + b.replace("'", "''").lower() + "'" for b in selected_behaviours)
    behaviour_clause = f"  AND LOWER(label) IN ({beh_in_list})\n"

# --- 6) SQL (chronological: ASC) ---
sql = f"""
SELECT time, confidence, label, sheep_id
FROM sheep_behavior_pred
WHERE time >= TIMESTAMP '{start_iso}'
  AND time <= TIMESTAMP '{end_iso}'
{sheep_clause}{behaviour_clause}ORDER BY time ASC
LIMIT 1000;
"""

st.subheader("SQL")
st.code(sql, language="sql")

# --- 7) Run SQL (v3 client) and normalize to pandas ---
try:
    with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        result = client.query(sql)

    if isinstance(result, pd.DataFrame):
        df = result
    elif isinstance(result, pa.Table):
        df = result.to_pandas()
    elif isinstance(result, list) and result and isinstance(result[0], pa.Table):
        df = pd.concat([t.to_pandas() for t in result], ignore_index=True)
    else:
        df = pd.DataFrame(result)

    if df is None or df.empty:
        st.info("No rows returned. Try different IDs/behaviours or widen the date/time window.")
    else:
        # Ensure proper chronological order in UI as well
        if "time" in df.columns:
            df = df.sort_values("time", ascending=True)
        st.dataframe(df)
        if {"time", "confidence"}.issubset(df.columns):
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            st.line_chart(df.set_index("time")["confidence"])
except Exception as e:
    st.error("SQL query failed. Check URL/Org/Token/Database (bucket) and retention.")
    st.exception(e)
