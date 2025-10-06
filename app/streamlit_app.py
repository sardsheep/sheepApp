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




# --- Controls for PIE CHART ---
pie_mode = st.radio(
    "Pie chart basis",
    options=["Use behaviour filter", "Ignore behaviour filter (all behaviours)"],
    index=0,
)
show_pie = st.button("Show behaviour pie chart")

# NEW: size + font controls
pie_size = st.slider("Pie size (inches)", 2.0, 8.0, 3.5, 0.1)
pie_font = st.slider("Pie label font size", 6, 16, 9)



# --- Controls for PIE CHART ---
pie_mode = st.radio(
    "Pie chart basis",
    options=["Use behaviour filter", "Ignore behaviour filter (all behaviours)"],
    index=0,
    help="Choose whether the pie respects the current behaviour multiselect."
)
show_pie = st.button("Show behaviour pie chart")

# --- 6) SQL (chronological: ASC) ---
# Base SQL (no behaviour filter) - we will use this when ignoring the filter for the pie
base_sql = f"""
SELECT time, confidence, label, sheep_id
FROM sheep_behavior_pred
WHERE time >= TIMESTAMP '{start_iso}'
  AND time <= TIMESTAMP '{end_iso}'
{sheep_clause}ORDER BY time ASC
LIMIT 1000;
"""

# Main SQL (respects behaviour filter for table/line chart)
sql = f"""
SELECT time, confidence, label, sheep_id
FROM sheep_behavior_pred
WHERE time >= TIMESTAMP '{start_iso}'
  AND time <= TIMESTAMP '{end_iso}'
{sheep_clause}{behaviour_clause}ORDER BY time ASC
LIMIT 1000;
"""

st.subheader("SQL (main)")
st.code(sql, language="sql")

# --- 7) Run SQL (v3 client) and normalize to pandas ---
try:
    with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        result = client.query(sql)

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
        st.info("No rows returned. Try different IDs/behaviours or widen the date/time window.")
    else:
        # Ensure time is datetime and sort old -> new
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        if "time" in df.columns:
            df = df.sort_values("time", ascending=True)

        st.dataframe(df)

        # Line chart of confidence over time (if available)
        if {"time", "confidence"}.issubset(df.columns):
            st.line_chart(df.set_index("time")["confidence"])

        # --- PIE CHART (on demand) ---
        if show_pie:
            # Decide data source for the pie
            if pie_mode.startswith("Use"):
                pie_df = df  # use the current filtered dataframe
                st.caption("Pie chart uses the current behaviour filter.")
            else:
                # Re-query ignoring the behaviour filter
                st.caption("Pie chart ignores the behaviour filter (all behaviours).")
                with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
                    pie_result = client.query(base_sql)
                if isinstance(pie_result, pd.DataFrame):
                    pie_df = pie_result
                elif isinstance(pie_result, pa.Table):
                    pie_df = pie_result.to_pandas()
                elif isinstance(pie_result, list) and pie_result and isinstance(pie_result[0], pa.Table):
                    pie_df = pd.concat([t.to_pandas() for t in pie_result], ignore_index=True)
                else:
                    pie_df = pd.DataFrame(pie_result)

                if "time" in pie_df.columns and not pd.api.types.is_datetime64_any_dtype(pie_df["time"]):
                    pie_df["time"] = pd.to_datetime(pie_df["time"], errors="coerce", utc=True)

            # Build PIE if labels available
            if pie_df is not None and not pie_df.empty and "label" in pie_df.columns:
                labels_norm = pie_df["label"].astype(str).str.strip().str.lower()

                known = [
                    "flehmen", "grazing", "head-butting", "lying",
                    "mating", "running", "standing", "walking",
                ]
                counts = labels_norm.value_counts().reindex(known, fill_value=0)
                total = int(counts.sum())

                if total > 0:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(pie_size, pie_size), dpi=120)  # smaller figure
                    ax.pie(
                        counts.values,
                        labels=known,
                        autopct=lambda p: f"{p:.1f}%",
                        startangle=90,
                        textprops={"fontsize": pie_font},      # smaller text
                        pctdistance=0.8,                       # keep % closer to center
                    )
ax.axis("equal")
ax.set_title("Behaviour share (%) in selected window", fontsize=pie_font + 1)
st.pyplot(fig, use_container_width=False)  # don't stretch to full width

                    summary = pd.DataFrame({
                        "count": counts.astype(int),
                        "percent": (counts / total * 100).round(2),
                    })
                    st.dataframe(summary)
                else:
                    st.info("No behaviour labels found in the selected window to plot.")
            else:
                st.info("No data or 'label' column missing for pie chart.")
except Exception as e:
    st.error("SQL query or rendering failed. Check URL/Org/Token/Database (bucket), permissions, and query syntax.")
    st.exception(e)
