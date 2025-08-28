import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Load processed dataset
df = pipeline.run_pipeline("telecom_tower_usaged.json")  # after cleaning, clustering, anomalies, recs
plt.rcParams["font.family"] = "Segoe UI Emoji"
st.set_page_config(page_title="Telecom Tower Dashboard", layout="wide")

st.title("📡 Telecom Network Performance Dashboard")

# ---------------- OVERVIEW ----------------
st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Towers", df["tower_id"].nunique())
col2.metric("Avg Latency (s)", round(df["latency_sec"].mean(), 2))
col3.metric("Avg Download Speed (Mbps)", round(df["download_speed_mbps"].mean(), 2))
col4.metric("Avg Dropped Calls", round(df["dropped_calls"].mean(), 2))

st.markdown("---")

# ---------------- TIME-SERIES ----------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

st.header("📈 Time-Series Analysis")

# Select metric
metric = st.selectbox("Choose metric:", ["latency_sec", "download_speed_mbps", "dropped_calls"])

# Date range filter
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.date_input(
    "📅 Select time range:",
    value=[min_date, max_date],  # default full range
    min_value=min_date,
    max_value=max_date
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
else:
    df_filtered = df.copy()

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=df_filtered, x="timestamp", y=metric, hue="operator", ax=ax)
ax.set_title(f"{metric} Over Time by Operator ({start_date} → {end_date})")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("---")

# ---------------- CLUSTERING ----------------
st.header("🔎 Clustering & Anomaly Detection")

fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", style="anomaly",
                palette="Set2", markers={1:"o", -1:"X"}, ax=ax)
ax.set_title("Tower Clusters with Anomalies")
st.pyplot(fig)

st.write("**Cluster Summary**")
st.dataframe(df.groupby("cluster")[["latency_sec","download_speed_mbps","tower_load_percent","dropped_calls"]].mean())

st.markdown("---")

# ---------------- MAP ----------------
st.header("🌍 Geospatial View of Towers")


with open("tower_map.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Render in Streamlit
st.components.v1.html(html_content, height=600, scrolling=True)

df["rec_category"] = df["recommendation"].apply(lambda x: x.split("→")[0])

st.subheader("📊 Recommendation Categories")
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(data=df, y="rec_category", order=df["rec_category"].value_counts().index, ax=ax)
ax.set_title("Most Common Issues Across Towers")
st.pyplot(fig)


st.title("📡 Tower Recommendations Dashboard")

# Search input
search_term = st.text_input("🔍 Search by Tower ID or Operator:")

if search_term:
    # Filter by tower_id (exact or partial match) OR operator name
    df_filtered = df[
        df["tower_id"].astype(str).str.contains(search_term, case=False, na=False)
        | df["operator"].str.contains(search_term, case=False, na=False)
    ]
else:
    df_filtered = df  # show all if no search


def generate_pdf(df_filtered):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.white,
        backColor=colors.HexColor("#1717a1"),
        leading=14,
        borderPadding=(8, 8, 8),
        spaceAfter=10,
    )
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.black,
        backColor=colors.HexColor("#ffcccc"),
        leading=14,
        borderPadding=(8, 8, 8),
        spaceAfter=10,
    )

    for _, row in df_filtered.iterrows():
        style = warning_style if "⚠️" in row["recommendation"] or "🚨" in row["recommendation"] else normal_style
        text = f"<b>Tower {row['tower_id']} ({row['operator']} - {row['network_type']})</b><br/>{row['recommendation']}"
        elements.append(Paragraph(text, style))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# Show recommendations
# Show recommendations (limited view if no search applied)
if df_filtered.empty:
    st.warning("No towers found matching your search.")
else:
    # If user did NOT search, only show first 5 towers
    if not search_term:
        df_to_display = df_filtered.head(5)
        st.info("Showing first 5 towers. Use search to see specific towers.")
    else:
        df_to_display = df_filtered

    # PDF download for displayed results
    pdf_buffer = generate_pdf(df_to_display)
    st.download_button(
        label="📥 Download Filtered Recommendations as PDF",
        data=pdf_buffer,
        file_name="filtered_recommendations.pdf",
        mime="application/pdf",
    )

    # Show recommendations on screen
    for _, row in df_to_display.iterrows():
        st.markdown(f"""
        <div style="padding:10px; margin:8px; border-radius:10px; 
                    background-color:{'#ffcccc' if '⚠️' in row['recommendation'] or '🚨' in row['recommendation'] else "#1717a1"}; color:white;">
            <b>Tower {row['tower_id']} ({row['operator']} - {row['network_type']})</b><br>
            {row['recommendation']}
        </div>
        """, unsafe_allow_html=True)


    # PDF download for filtered results
    

st.header("📖 Network Performance Story")

story = """
Our analysis of telecom tower usage reveals some interesting patterns:

- **5G towers** generally perform well, but a few outliers show speeds as low as **10 Mbps**, behaving more like 3G towers. These anomalies likely indicate **faulty configuration or backhaul congestion**.  
- In **4G/LTE towers**, several sites report **latencies above 0.7 seconds**, far beyond the expected 50–100 ms range. Such towers contribute to **customer complaints about call drops and slow browsing**.  
- Towers with **>85% load** consistently show **reduced speeds** and **increased dropped calls**, proving that **overloaded towers degrade service quality**.  
- Geospatial mapping highlights that rural towers are more prone to **packet loss (>10%)**, suggesting weaker infrastructure in remote regions.  
- The clustering analysis divided towers into groups: **Healthy High-Speed Towers, Overloaded Towers, Latency-Prone Towers, and Old Maintenance-Due Towers**.  
- Finally, our **automated recommendation engine** provides actionable insights — from **load balancing upgrades** to **dispatching engineers** for anomaly-prone towers.  

This story demonstrates how raw operational data can be transformed into **business decisions** that directly improve **network reliability and customer experience**.
"""

st.markdown(story)

