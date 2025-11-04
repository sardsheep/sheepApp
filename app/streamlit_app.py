import streamlit as st
import pandas as pd
import pyarrow as pa
from influxdb_client import InfluxDBClient            # v2 client (health)
from influxdb_client_3 import InfluxDBClient3         # v3 client (SQL)
from datetime import datetime, timedelta, timezone, time as dtime

try:
    from zoneinfo import ZoneInfo
    TZ_LOCAL = ZoneInfo("Europe/Rome")
except Exception:
    TZ_LOCAL = timezone.utc


# --- Debug toggle ---
SHOW_DEBUG = False

st.set_page_config(page_title="Sheep Behavior — SQL", layout="wide")
st.title("🐑 Sheep Behavior")


# --- 1) Secrets ---
try:
    cfg = st.secrets["influx"]
    URL = cfg["url"]
    TOKEN = cfg["token"]
    ORG = cfg["org"]
    DB = cfg["bucket"]
except Exception as e:
    st.error("Missing or malformed secrets. Set [influx] url/token/org/bucket in Streamlit Secrets.")
    st.exception(e)
    st.stop()

if SHOW_DEBUG:
    with st.expander("Connection config (sanitized)"):
        st.write({
            "url": URL,
            "org": ORG,
            "database/bucket": DB,
            "token_prefix": TOKEN[:6] + "..." if isinstance(TOKEN, str) else "invalid",
        })


# --- 2) Sheep type filter ---
RAM_IDS = ["1", "2", "3", "4", "5"]
EWE_IDS = ["6", "7", "8", "9", "10"]

sheep_type_choice = st.radio(
    "Sheep type",
    options=["Any", "Ram", "Ewe"],
    index=0,
    horizontal=True,
)

type_clause = ""
if sheep_type_choice == "Ram":
    id_list = ",".join(f"'{sid}'" for sid in RAM_IDS)
    type_clause = f"  AND sheep_id IN ({id_list})\n"
elif sheep_type_choice == "Ewe":
    id_list = ",".join(f"'{sid}'" for sid in EWE_IDS)
    type_clause = f"  AND sheep_id IN ({id_list})\n"


# --- 3) Time window ---
today_local = datetime.now(TZ_LOCAL).date()
default_start_date = today_local - timedelta(days=30)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start date", value=default_start_date)
    start_time = st.time_input("Start time", value=dtime(0, 0))
with c2:
    end_date = st.date_input("End date", value=today_local)
    end_time = st.time_input("End time", value=dtime(23, 59))

start_dt_local = datetime.combine(start_date, start_time, tzinfo=TZ_LOCAL)
end_dt_local = datetime.combine(end_date, end_time, tzinfo=TZ_LOCAL)

if start_dt_local > end_dt_local:
    st.error("Start datetime must be before end datetime.")
    st.stop()

start_iso = start_dt_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
end_iso = end_dt_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

st.caption(
    f"Querying from **{start_dt_local.strftime('%Y-%m-%d %H:%M %Z')}** "
    f"to **{end_dt_local.strftime('%Y-%m-%d %H:%M %Z')}** "
    f"(UTC: {start_iso} → {end_iso})"
)


# --- 4) Sheep ID filter ---
ALL_SHEEP_IDS = [str(i) for i in range(1, 11)]
selected_ids = st.multiselect(
    "Sheep IDs (optional)",
    options=ALL_SHEEP_IDS,
    default=[],
    help="Pick one or more IDs (1–10). Leave empty to include all.",
)
if selected_ids:
    id_in_list = ",".join(f"'{sid}'" for sid in selected_ids)
else:
    id_in_list = ",".join(f"'{sid}'" for sid in ALL_SHEEP_IDS)
sheep_clause = f"  AND sheep_id IN ({id_in_list})\n"


# --- 5) Behaviour filter ---
ALL_BEHAVIOURS = [
    "flehmen", "grazing", "head-butting", "lying",
    "mating", "running", "standing", "walking",
]
selected_behaviours = st.multiselect(
    "Predicted Behaviour (optional filter)",
    options=ALL_BEHAVIOURS,
    default=[],
    help="Pick one or more behaviours. Leave empty to include all.",
)

behaviour_clause = ""
if selected_behaviours:
    beh_in_list = ",".join("'" + b.lower().strip() + "'" for b in selected_behaviours)
    behaviour_clause = f"  AND LOWER(TRIM(label)) IN ({beh_in_list})\n"


# --- 6) SQL query builder ---
ROW_LIMIT = 500_000
st.caption(f"Fetching up to **{ROW_LIMIT:,}** rows.")

def build_sql(include_type: bool, include_behaviour: bool) -> str:
    return f"""
SELECT time, confidence, label, sheep_id{(", type" if include_type else "")}
FROM sheep_behavior_pred
WHERE time >= TIMESTAMP '{start_iso}'
  AND time <= TIMESTAMP '{end_iso}'
{sheep_clause}{(type_clause if include_type else "")}{(behaviour_clause if include_behaviour else "")}
ORDER BY time ASC
LIMIT {ROW_LIMIT};
"""

sql_current = build_sql(include_type=(type_clause != ""), include_behaviour=True)


# --- 7) Run query and show table ---
try:
    with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        result = client.query(sql_current)

    if isinstance(result, pd.DataFrame):
        df = result
    elif isinstance(result, pa.Table):
        df = result.to_pandas()
    elif isinstance(result, list) and result and isinstance(result[0], pa.Table):
        df = pd.concat([t.to_pandas() for t in result], ignore_index=True)
    else:
        df = pd.DataFrame(result)

    if df.empty:
        st.info("No rows returned.")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        st.dataframe(df)






        # --- BEHAVIOUR OVER TIME (Altair line chart) ---
        st.subheader("Behaviour occurrences over time (seconds)")

        if {"time", "label", "sheep_id"}.issubset(df.columns):
            import altair as alt

            behaviour_for_line = st.selectbox(
                "Choose behaviour to plot",
                options=ALL_BEHAVIOURS,
                index=ALL_BEHAVIOURS.index("walking") if "walking" in ALL_BEHAVIOURS else 0,
                key="behaviour_line_select",
            )

            df_line = df.copy()
            df_line["label_norm"] = df_line["label"].astype(str).str.strip().str.lower()
            df_line["time_sec"] = df_line["time"].dt.floor("S")
            target = behaviour_for_line.lower().strip()

            events = (
                df_line.assign(value=(df_line["label_norm"] == target).astype(int))
                .groupby(["time_sec", "sheep_id"], as_index=False)["value"]
                .max()
                .query("value == 1")
            )

            if events.empty:
                st.info("No occurrences of the selected behaviour.")
            else:
                domain_min = df["time"].min()
                domain_max = df["time"].max()
                zoom_x = alt.selection_interval(bind='scales', encodings=['x'])

                chart = (
                    alt.Chart(events)
                    .mark_circle(size=40, opacity=0.9)
                    .encode(
                        x=alt.X(
                            "time_sec:T",
                            title="Time",
                            scale=alt.Scale(domain=[domain_min, domain_max]),
                        ),
                        y=alt.Y(
                            "value:Q",
                            title="Occurrence",
                            scale=alt.Scale(domain=[0, 1.1]),
                            axis=alt.Axis(values=[1], labels=False, ticks=False),
                        ),
                        color=alt.Color("sheep_id:N", title="Sheep ID"),
                        tooltip=[
                            alt.Tooltip("time_sec:T", title="Time"),
                            alt.Tooltip("sheep_id:N", title="Sheep"),
                        ],
                    )
                    .properties(height=240)
                    .add_params(zoom_x)
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption("Tip: scroll to zoom, drag to pan, double-click to reset.")
        else:
            st.info("Missing columns for behaviour chart.")


        # --- BEHAVIOUR DISTRIBUTION (Pie chart) ---
        st.subheader("Behaviour distribution (pie)")
        import matplotlib.pyplot as plt

        labels_norm = df["label"].astype(str).str.strip().str.lower()
        known = [
            "flehmen", "grazing", "head-butting", "lying",
            "mating", "running", "standing", "walking",
        ]
        counts = labels_norm.value_counts().reindex(known, fill_value=0)
        total = int(counts.sum())

        if total > 0:
            fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=120)
            wedges, texts, autotexts = ax.pie(
                counts.values,
                startangle=90,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 3.0 else "",
                pctdistance=0.72,
                textprops={"fontsize": 9},
            )
            ax.axis("equal")
            ax.set_title("Behaviour share (%)", fontsize=10)

            legend_labels = [
                f"{name} — {int(cnt)} ({(cnt/total*100):.1f}%)"
                for name, cnt in zip(known, counts.values)
            ]
            ax.legend(
                wedges, legend_labels, title="Behaviour",
                loc="center left", bbox_to_anchor=(1.0, 0.5),
                fontsize=9, title_fontsize=10, frameon=False,
            )

            fig.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=False)
        else:
            st.info("No behaviour labels found for pie chart.")










except Exception as e:
    st.error("Failed to query InfluxDB.")
    st.exception(e)
    st.stop()










# --- 8) Chat with AI ---
st.header("💬 Chat with the AI")

from openai import OpenAI

def generate_query_from_prompt(prompt: str) -> str | None:
    """
    Generate an InfluxDB SQL query based on a natural language question.
    Designed for sheep behavior analysis (time, label, confidence, sheep_id, type).
    """

    if not prompt or not isinstance(prompt, str):
        return None

    p = prompt.lower().strip()

    # ---- Behavior timing ----
    if "first" in p and "flehmen" in p:
        return "SELECT MIN(time) AS first_flehmen_time FROM sheep_behavior_pred WHERE LOWER(label)='flehmen'"
    if "last" in p and "flehmen" in p:
        return "SELECT MAX(time) AS last_flehmen_time FROM sheep_behavior_pred WHERE LOWER(label)='flehmen'"
    if "first" in p and "running" in p:
        return "SELECT MIN(time) AS first_running_time FROM sheep_behavior_pred WHERE LOWER(label)='running'"
    if "last" in p and "running" in p:
        return "SELECT MAX(time) AS last_running_time FROM sheep_behavior_pred WHERE LOWER(label)='running'"
    if "first" in p and "standing" in p:
        return "SELECT MIN(time) AS first_standing_time FROM sheep_behavior_pred WHERE LOWER(label)='standing'"
    if "last" in p and "standing" in p:
        return "SELECT MAX(time) AS last_standing_time FROM sheep_behavior_pred WHERE LOWER(label)='standing'"

    # ---- Frequency / counts ----
    if "count" in p or "how many" in p or "number of" in p:
        if "each" in p or "per behavior" in p or "per behaviour" in p:
            return "SELECT label, COUNT(*) AS count FROM sheep_behavior_pred GROUP BY label ORDER BY count DESC"
        if "sheep" in p:
            return "SELECT sheep_id, COUNT(*) AS count FROM sheep_behavior_pred GROUP BY sheep_id ORDER BY count DESC"
        if "flehmen" in p:
            return "SELECT COUNT(*) AS flehmen_count FROM sheep_behavior_pred WHERE LOWER(label)='flehmen'"
        if "running" in p:
            return "SELECT COUNT(*) AS running_count FROM sheep_behavior_pred WHERE LOWER(label)='running'"
        return "SELECT label, COUNT(*) AS count FROM sheep_behavior_pred GROUP BY label ORDER BY count DESC"

    # ---- Most active / least active ----
    if "most active" in p or "most events" in p:
        return "SELECT sheep_id, COUNT(*) AS events FROM sheep_behavior_pred GROUP BY sheep_id ORDER BY events DESC LIMIT 1"
    if "least active" in p or "inactive" in p:
        return "SELECT sheep_id, COUNT(*) AS events FROM sheep_behavior_pred GROUP BY sheep_id ORDER BY events ASC LIMIT 1"

    # ---- Behavior proportions ----
    if "percentage" in p or "proportion" in p or "share" in p:
        return """
        SELECT label,
               COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sheep_behavior_pred) AS percentage
        FROM sheep_behavior_pred
        GROUP BY label
        ORDER BY percentage DESC
        """


        # ---- Average confidence ----
    if "average confidence" in p or "mean confidence" in p:
        # Specific behavior, e.g. "average confidence for lying"
        import re
        match = re.search(r"for ([a-z\-]+)", p)
        if match:
            behavior = match.group(1).lower()
            return f"SELECT AVG(confidence) AS avg_conf_{behavior} FROM sheep_behavior_pred WHERE LOWER(label)='{behavior}'"
    
        # Per-behavior average
        if "each" in p or "per behavior" in p or "per behaviour" in p:
            return "SELECT label, AVG(confidence) AS avg_conf FROM sheep_behavior_pred GROUP BY label ORDER BY avg_conf DESC"
    
        # Ram / Ewe specific
        if "ram" in p:
            return "SELECT AVG(confidence) AS avg_conf_ram FROM sheep_behavior_pred WHERE LOWER(type)='ram'"
        if "ewe" in p:
            return "SELECT AVG(confidence) AS avg_conf_ewe FROM sheep_behavior_pred WHERE LOWER(type)='ewe'"
    
        # Default overall
        return "SELECT AVG(confidence) AS avg_conf_overall FROM sheep_behavior_pred"

    # ---- Compare rams vs ewes ----
    if "ram" in p and "ewe" in p:
        return """
        SELECT type, COUNT(*) AS events
        FROM sheep_behavior_pred
        WHERE LOWER(type) IN ('ram', 'ewe')
        GROUP BY type
        ORDER BY events DESC
        """

    # ---- Time-range queries ----
    if "between" in p and "and" in p:
        # Extract potential timestamps (very basic matching)
        import re
        times = re.findall(r"\d{2}:\d{2}", p)
        if len(times) == 2:
            return f"SELECT * FROM sheep_behavior_pred WHERE time >= TIMESTAMP '2025-07-04T{times[0]}:00Z' AND time <= TIMESTAMP '2025-07-04T{times[1]}:00Z' ORDER BY time ASC LIMIT 1000"
        return "SELECT * FROM sheep_behavior_pred ORDER BY time ASC LIMIT 1000"

    # ---- Behavior transitions ----
    if "after" in p and "flehmen" in p:
        return """
        SELECT * FROM sheep_behavior_pred
        WHERE time > (SELECT MAX(time) FROM sheep_behavior_pred WHERE LOWER(label)='flehmen')
        ORDER BY time ASC LIMIT 10
        """
    if "before" in p and "mating" in p:
        return """
        SELECT * FROM sheep_behavior_pred
        WHERE time < (SELECT MIN(time) FROM sheep_behavior_pred WHERE LOWER(label)='mating')
        ORDER BY time DESC LIMIT 10
        """

    # ---- Default summaries ----
    if "summary" in p or "overview" in p:
        return "SELECT label, COUNT(*) AS count, AVG(confidence) AS avg_conf FROM sheep_behavior_pred GROUP BY label ORDER BY count DESC"

    # ---- Last events ----
    if "latest" in p or "most recent" in p:
        return "SELECT * FROM sheep_behavior_pred ORDER BY time DESC LIMIT 5"

    # ---- Fallback ----
    return None



try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
    model_name = "llama-3.1-8b-instant"
except Exception:
    st.error("Missing Groq API key.")
    st.stop()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": (
            "You are an assistant that analyzes sheep behavior data stored in InfluxDB. "
            "The dataset includes: time, sheep_id, label, confidence, and optionally type (Ram or Ewe). "
            "Use the context provided to answer factual questions accurately."
        )}
    ]

for m in st.session_state.chat_messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about sheep behavior 🐑")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Query result placeholder ---
    query_result_text = ""
    query = generate_query_from_prompt(prompt)
    if query:
        try:
            with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as qclient:
                qres = qclient.query(query)
                if isinstance(qres, pd.DataFrame):
                    qdf = qres
                elif isinstance(qres, pa.Table):
                    qdf = qres.to_pandas()
                elif isinstance(qres, list) and qres and isinstance(qres[0], pa.Table):
                    qdf = pd.concat([t.to_pandas() for t in qres], ignore_index=True)
                else:
                    qdf = pd.DataFrame(qres)

            if not qdf.empty:
                query_result_text = f"### Query result from InfluxDB\nQuery: `{query}`\n\n{qdf.to_string(index=False)}"
            else:
                query_result_text = f"### Query result from InfluxDB\nQuery: `{query}`\n\n(No rows returned)"
        except Exception as e:
            query_result_text = f"Error executing query: {e}"

    # --- Dataset context ---
    if 'df' in locals() and not df.empty:
        df_sorted = df.sort_values("time", ascending=True).copy()
        df_sorted["label_norm"] = df_sorted["label"].astype(str).str.lower()
        summary = df_sorted["label_norm"].value_counts().to_string()
        context_summary = (
            f"### Dataset summary:\n"
            f"Rows: {len(df_sorted):,}\n"
            f"Time range: {df_sorted['time'].min()} → {df_sorted['time'].max()}\n"
            f"**Behavior counts:**\n{summary}"
        )
    else:
        context_summary = "No dataset loaded from InfluxDB."

    # --- Combine both safely ---
    context_msg = {
        "role": "system",
        "content": query_result_text or context_summary
    }

    messages = st.session_state.chat_messages + [context_msg]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Chat error: {e}"

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # --- Health check ---
    try:
        with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
            health = client.health()
            st.success(f"Influx health: {health.status} — {health.message}")
    except Exception as e:
        st.error("Could not reach InfluxDB.")
        st.exception(e)

