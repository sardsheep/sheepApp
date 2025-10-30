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




# --- Debug toggles ---
SHOW_DEBUG = False  # set True if you want to see connection info / health


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

# Only show when debugging
if SHOW_DEBUG:
    with st.expander("Connection config (sanitized)"):
        st.write({
            "url": URL,
            "org": ORG,
            "database/bucket": DB,
            "token_prefix": TOKEN[:6] + "..." if isinstance(TOKEN, str) and len(TOKEN) > 6 else "short/invalid",
        })








# --- 5b) Sheep type radio (Ram/Ewe) + SQL clause (safe fallback if column missing) ---

type_clause = ""
sheep_type_choice = st.radio(
    "Sheep type",
    options=["Any", "Ram", "Ewe"],
    index=0,
    horizontal=True,
    key="sheep_type_radio",
)

# Define which IDs correspond to Rams vs Ewes
RAM_IDS  = ["1", "2", "3", "4", "5"]
EWE_IDS  = ["6", "7", "8", "9", "10"]


# Build clause based on selection
type_clause = ""
if sheep_type_choice == "Ram":
    id_list = ",".join(f"'{sid}'" for sid in RAM_IDS)
    type_clause = f"  AND sheep_id IN ({id_list})\n"
elif sheep_type_choice == "Ewe":
    id_list = ",".join(f"'{sid}'" for sid in EWE_IDS)
    type_clause = f"  AND sheep_id IN ({id_list})\n"




# --- 3) Inputs: explicit start/end date & time (local) ---
today_local = datetime.now(TZ_LOCAL).date()
default_start_date = today_local - timedelta(days=30)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start date (local)", value=default_start_date, key="start_date")
    start_time = st.time_input("Start time (local)", value=dtime(0, 0), key="start_time")
with c2:
    end_date = st.date_input("End date (local)", value=today_local, key="end_date")
    end_time = st.time_input("End time (local)", value=dtime(23, 59), key="end_time")

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
    help="Pick one or more IDs (1–10). Leave empty to include all 10.",
    key="sheep_ids"
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
    help="Pick one or more behaviours. Leave empty to include all.",
    key="behaviours"
)

behaviour_clause = ""
if selected_behaviours:
    beh_in_list = ",".join("'" + b.replace("'", "''").lower().strip() + "'" for b in selected_behaviours)
    behaviour_clause = f"  AND LOWER(TRIM(label)) IN ({beh_in_list})\n"



# --- 6) SQL (chronological: ASC) ---
ROW_LIMIT = 500_000  # fixed default
st.caption(f"Fetching up to **{ROW_LIMIT:,}** rows.")

def build_sql(include_type: bool, include_behaviour: bool, limit: int = ROW_LIMIT) -> str:
    return f"""
SELECT time, confidence, label, sheep_id{(", type" if include_type else "")}
FROM sheep_behavior_pred
WHERE time >= TIMESTAMP '{start_iso}'
  AND time <= TIMESTAMP '{end_iso}'
{sheep_clause}{(type_clause if include_type else "")}{(behaviour_clause if include_behaviour else "")}ORDER BY time ASC
LIMIT {limit};
"""

# Build both strings with current choices
base_sql_current = build_sql(include_type=(type_clause != ""), include_behaviour=False)
sql_current      = build_sql(include_type=(type_clause != ""), include_behaviour=True)

# --- 7) Run SQL (v3 client) and normalize to pandas ---
try:
    with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
        # Try with sheep_type filter (if requested). If the column doesn't exist, retry without it.
        try:
            result = client.query(sql_current)
            type_filter_applied = (type_clause != "")
        except Exception as e1:
            msg = str(e1).lower()
            if (type_clause != "") and ("sheep_type" in msg or "column" in msg and ("not" in msg and ("exist" in msg or "found" in msg))):
                st.warning("Column `sheep_type` not found — ignoring Ram/Ewe filter.")
                # Rebuild SQLs without the type clause and retry
                base_sql_current = build_sql(include_type=False, include_behaviour=False)
                sql_current      = build_sql(include_type=False, include_behaviour=True)
                result = client.query(sql_current)
                type_filter_applied = False
            else:
                raise  # not a missing-column error; bubble up

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

        # --- TABLE ---
        st.dataframe(df)




        
        # --- BEHAVIOUR OVER TIME (events-only points; interactive zoom) ---
        st.subheader("Behaviour occurrences over time (seconds)")
        if {"time", "label", "sheep_id"}.issubset(df.columns):
            behaviour_for_line = st.selectbox(
                "Choose behaviour to plot",
                options=ALL_BEHAVIOURS,
                index=ALL_BEHAVIOURS.index("walking") if "walking" in ALL_BEHAVIOURS else 0,
                key="behaviour_line_select",
            )

            # Auto-fallback: if the chosen behaviour isn't in the multiselect filter, query without it
            need_fallback = (
                bool(selected_behaviours)
                and behaviour_for_line.lower().strip() not in [b.lower().strip() for b in selected_behaviours]
            )

            if need_fallback:
                st.info(
                    f"Using all behaviours for the chart because '{behaviour_for_line}' "
                    f"is not in the behaviour filter above."
                )
                with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
                    line_result = client.query(base_sql_current)
                if isinstance(line_result, pd.DataFrame):
                    line_df = line_result
                elif isinstance(line_result, pa.Table):
                    line_df = line_result.to_pandas()
                elif isinstance(line_result, list) and line_result and isinstance(line_result[0], pa.Table):
                    line_df = pd.concat([t.to_pandas() for t in line_result], ignore_index=True)
                else:
                    line_df = pd.DataFrame(line_result)
            else:
                line_df = df.copy()

            # Normalize time + labels
            if "time" in line_df.columns and not pd.api.types.is_datetime64_any_dtype(line_df["time"]):
                line_df["time"] = pd.to_datetime(line_df["time"], errors="coerce", utc=True)

            if line_df.empty:
                st.info("No data available to plot.")
            else:
                plot = line_df.copy()
                plot["time"]      = pd.to_datetime(plot["time"], utc=True)
                plot["time_sec"]  = plot["time"].dt.floor("S")
                plot["sheep_id"]  = plot["sheep_id"].astype(str)
                plot["label_norm"]= plot["label"].astype(str).str.strip().str.lower()
                target = behaviour_for_line.lower().strip()

                # Keep only seconds where the chosen behaviour occurred (points at y=1)
                events = (
                    plot.assign(value=(plot["label_norm"] == target).astype(int))
                        .groupby(["time_sec", "sheep_id"], as_index=False)["value"].max()
                        .query("value == 1")
                )

                if events.empty:
                    st.info("No occurrences of the selected behaviour in this window/IDs.")
                else:
                    try:
                        import altair as alt
                        # Force x-axis to full selected window; allow zoom/pan on x
                        domain_min = pd.to_datetime(start_iso, utc=True)
                        domain_max = pd.to_datetime(end_iso,   utc=True)
                        zoom_x = alt.selection_interval(bind='scales', encodings=['x'])

                        chart = (
                            alt.Chart(events)
                            .mark_circle(size=40, opacity=0.9)
                            .encode(
                                x=alt.X(
                                    "time_sec:T",
                                    title="Time",
                                    scale=alt.Scale(domain=[domain_min, domain_max])
                                ),
                                y=alt.Y(
                                    "value:Q",
                                    title="Occurrence",
                                    scale=alt.Scale(domain=[0, 1.1]),
                                    axis=alt.Axis(values=[1], labels=False, ticks=False)
                                ),
                                color=alt.Color("sheep_id:N", title="Sheep ID"),
                                tooltip=[alt.Tooltip("time_sec:T", title="Time"), alt.Tooltip("sheep_id:N", title="Sheep")]
                            )
                            .properties(height=240)
                            .add_params(zoom_x)
                        )
                        st.altair_chart(chart, use_container_width=True)
                        st.caption("Tip: scroll to zoom, drag to pan, double-click to reset.")
                    except Exception:
                        # Fallback: sparse wide frame with NaNs (no zeros), still spans full window
                        wide = events.pivot(index="time_sec", columns="sheep_id", values="value").sort_index()
                        full_idx = pd.date_range(start=pd.to_datetime(start_iso, utc=True),
                                                 end=pd.to_datetime(end_iso,   utc=True),
                                                 freq="1S", tz="UTC")
                        wide = wide.reindex(full_idx)  # NaNs where no event -> no connecting line
                        st.line_chart(wide)
        else:
            st.info("Required columns ('time', 'label', 'sheep_id') are missing; cannot draw the behaviour chart.")

        # --------------- PIE CONTROLS & CHART (AFTER TABLE & LINE) ---------------
        st.divider()
        st.subheader("Behaviour distribution (pie)")

        pie_mode = st.radio(
            "Pie chart basis",
            options=["Use behaviour filter", "Ignore behaviour filter (all behaviours)"],
            index=0,
            help="Choose whether the pie respects the current behaviour multiselect.",
            key="pie_mode_radio",
        )
        show_pie = st.button("Show behaviour pie chart", key="show_pie_btn")

        if show_pie:
            # Decide data source for the pie
            if pie_mode.startswith("Use"):
                pie_df = df  # use the current filtered dataframe
                st.caption("Pie chart uses the current behaviour filter.")
            else:
                st.caption("Pie chart ignores the behaviour filter (all behaviours).")
                with InfluxDBClient3(host=URL, token=TOKEN, org=ORG, database=DB) as client:
                    pie_result = client.query(base_sql_current)
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

                    # No overlap: show % only for larger slices; names in legend
                    MIN_PCT_LABEL = 3.0
                    def _fmt_autopct(pct):
                        return f"{pct:.1f}%" if pct >= MIN_PCT_LABEL else ""

                    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=120)
                    wedges, texts, autotexts = ax.pie(
                        counts.values,
                        startangle=90,
                        autopct=_fmt_autopct,
                        pctdistance=0.72,
                        labels=None,
                        textprops={"fontsize": 9},
                    )
                    ax.axis("equal")
                    ax.set_title("Behaviour share (%) in selected window", fontsize=10)

                    legend_labels = [
                        f"{name} — {int(cnt)} ({(cnt/total*100):.1f}%)"
                        for name, cnt in zip(known, counts.values)
                    ]
                    ax.legend(
                        wedges, legend_labels, title="Behaviour",
                        loc="center left", bbox_to_anchor=(1.0, 0.5),
                        fontsize=9, title_fontsize=10, frameon=False
                    )

                    fig.tight_layout(pad=0.4)
                    st.pyplot(fig, use_container_width=False)
                else:
                    st.info("No behaviour labels found in the selected window to plot.")
            else:
                st.info("No data or 'label' column missing for pie chart.")
except Exception as e:
    st.error("SQL query or rendering failed. Check URL/Org/Token/Database (bucket), permissions, and query syntax.")
    st.exception(e)
















# --- Simple LLM Chat (Groq cloud API + InfluxDB context) ---
st.header("💬 Chat with the AI")

from openai import OpenAI

# Try to connect to Groq (free cloud API)
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
    model_name = "llama-3.1-8b-instant"  # fast, cost-effective model
except Exception:
    st.error("Missing Groq API key. Add it in Secrets as [groq].api_key")
    st.stop()

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": (
            "You are an assistant that analyzes sheep behavior data stored in InfluxDB. "
            "The dataset includes: time, sheep_id, label, confidence, and optionally type (Ram or Ewe). "
            "When answering, use the actual dataset summary provided below. "
            "You can answer questions like 'Which behavior is most frequent?', "
            "'Which sheep type (Ram/Ewe) is more active?', or 'Average confidence per behavior'."
        )}
    ]

# Display chat history (skip system message)
for m in st.session_state.chat_messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])









# --- Chat input ---
prompt = st.chat_input("Ask something about sheep behavior 🐑")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Build extended AI context with full temporal info ---
    if 'df' in locals() and df is not None and not df.empty:
        # Normalize time/labels
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df_sorted = df.sort_values("time", ascending=True).copy()
        df_sorted["label"] = df_sorted["label"].astype(str)
        df_sorted["label_norm"] = df_sorted["label"].str.strip().str.lower()

        cols = [c for c in ['sheep_id', 'label', 'confidence', 'type', 'time'] if c in df_sorted.columns]

        # Limit for LLM context size (raise if your dataset is small)
        MAX_ROWS_FOR_AI = 2000
        timeline_df = df_sorted[["time", "sheep_id", "label_norm"]].head(MAX_ROWS_FOR_AI)
        timeline = timeline_df.to_string(index=False)

        time_min = df_sorted["time"].min()
        time_max = df_sorted["time"].max()
        label_counts = (
            df_sorted["label_norm"].value_counts()
            .sort_values(ascending=False)
            .to_string()
        )

        context_summary = (
            f"### Dataset context from InfluxDB:\n"
            f"- Total rows: {len(df_sorted):,}\n"
            f"- Time range: {time_min} → {time_max}\n"
            f"- Columns available: {', '.join(cols)}\n\n"
            f"**Behavior frequency summary:**\n{label_counts}\n\n"
            f"**Timeline (first {len(timeline_df)} time–sheep–behaviour entries):**\n"
            f"{timeline}"
        )
    else:
        context_summary = (
            "No recent data available from InfluxDB. "
            "Try running a query or widening your date/time range."
        )

    # Keep this INSIDE the if prompt: block to avoid IndentationError
    with st.expander("🔍 AI Context Preview", expanded=False):
        st.text(context_summary)

    # Add context as an extra system message
    context_msg = {"role": "system", "content": context_summary}
    messages = st.session_state.chat_messages + [context_msg]

    # --- Call Groq LLM ---
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Chat error: {e}"

    # --- Display answer ---
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    











    
    # --- Optional: Show what AI sees ---
    with st.expander("🔍 AI Context Preview", expanded=False):
        st.text(context_summary)

    # Add context as an extra system message
    context_msg = {"role": "system", "content": context_summary}
    messages = st.session_state.chat_messages + [context_msg]

    # --- Call Groq LLM ---
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Chat error: {e}"

    # --- Display answer ---
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
















# --- 2) Connectivity check (v2 health) ---
try:
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        health = client.health()
        st.success(f"Influx health: {health.status} — {health.message}")
except Exception as e:
    st.error("Could not reach InfluxDB. Check URL (region!), org, and token scope.")
    st.exception(e)
    st.stop()
