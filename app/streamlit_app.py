import streamlit as st
import pandas as pd
import pyarrow as pa
from influxdb_client import InfluxDBClient            # v2 client (health)
from influxdb_client_3 import InfluxDBClient3         # v3 client (SQL)

from datetime import datetime, date, time as dtime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    TZ_LOCAL = ZoneInfo("Europe/Rome")  # adjust if needed
except Exception:
    TZ_LOCAL = timezone.utc  # fallback

st.set_page_config(page_title="Sheep Behavior — SQL", layout="wide")
st.title("🐑 Sheep Behavior")

# --- 1) Secrets ---
try:
    cfg = st.secrets["influx"]
    URL   = str(cfg["url"]).strip()     # e.g., "https://eu-central-1-1.aws.cloud2.influxdata.com"
    TOKEN = str(cfg["token"]).strip()   # READ token
    ORG   = str(cfg["org"]).strip()     # org name or ID
    DB    = str(cfg["bucket"]).strip()  # SQL "database" == your bucket
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
        "org_repr": repr(ORG),  # useful to detect hidden whitespace/newlines
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

# --- 3) Inputs: Sheep ID + Date/Time range (local -> UTC) ---
sheep_id = st.text_input("Sheep ID", value="1")

today_local = datetime.now(TZ_LOCAL).date()
default_start = today_local - timedelta(days=30)

start_date, end_date = st.date_input(
    "Date range (local time)",
    value=(default_start, today_local),
    help="Range is interpreted in your local timezone and converted to UTC for the query."
)

start_time = st.time_input("Start time (local)", value=dtime(0, 0))
end_time   = st.time_input("End time (local)",   value=dtime(23, 59))

# Normalize/validate range
if start_date > end_date or (start_date == end_date and start_time > end_time):
    st.warning("Start is after end; swapping.")
    # Build datetime first, then swap if needed
    sdt = datetime.combine(start_date, start_time, tzinfo=TZ_LOCAL)
    edt = datetime.combine(end_date,   end_time,   tzinfo=TZ_LOCAL)
    if sdt > edt:
        sdt, edt = edt, sdt
else:
    sdt = datetime.combine(start_date, start_time, tzinfo=TZ_LOCAL)
    edt = datetime.combine(end_date,   end_time,   tzinfo=TZ_LOCAL)

# Convert to UTC ISO-8601 (Z)
start_iso = sdt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
end_iso   = edt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

st.caption(f"Querying from **{sdt.strftime('%Y-%m-%d %H:%M')} {sdt.tzname()}** "
           f"to **{edt.strftime('%Y-%m-%d %H:%M')} {edt.tzname()}** "
           f"(UTC: {start_iso} → {end_iso})")

# Minimal sanitization for sheep_id in SQL string literal
safe_sheep_id = sheep_id.replace("'", "''")

# --- 4) SQL ---
sql = f"""
SELECT time, confidence, label, sheep_id
FROM sheep_behavior_pred
WHERE sheep_id = '{safe_sheep_id}'
  AND time >= TIMESTAMP '{start_iso}'
  AND time <= TIMESTAMP '{end_iso}'
ORDER BY time DESC
LIMIT 1000;
"""

st.subheader("SQL")
st.code(sql, language="sql")

# --- 5) Run SQL (v3 client) and normalize to pandas ---
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
        df = pd.DataFrame(result)

    if df is None or df.empty:
        st.info("No rows returned. Try another Sheep ID, expand the date/time window, or confirm recent data exists.")
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
