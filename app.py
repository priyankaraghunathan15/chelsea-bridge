import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import random
from sklearn.preprocessing import StandardScaler

# Set Boston timezone
BOSTON_TZ = ZoneInfo("America/New_York")
now = datetime.now(BOSTON_TZ)

# Page configuration
st.set_page_config(page_title="Chelsea Bridge Lift Forecast System", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 600;
            padding: 12px 24px;
        }
        div[data-testid="column"] > div:hover {
            transform: scale(1.015);
            transition: all 0.2s ease-in-out;
            box-shadow: 3px 3px 12px rgba(0, 0, 0, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load("lightgbm_lift_model.pkl")
scaler = joblib.load("scaler.pkl")

# Simulated features (replace with live inputs if needed)
hour = now.hour
day_of_week = now.weekday()
is_weekend = int(day_of_week >= 5)
is_peak_hour = int(hour in range(7, 10) or hour in range(16, 19))
time_since_last_lift = 45

features = {
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "is_peak_hour": is_peak_hour,
    "time_since_last_lift": time_since_last_lift,
    "temp_max_f": 66,
    "temp_min_f": 55,
    "precip_in": 0.0,
    "snowfall_in": 0.0,
    "snow_depth_in": 0.0,
    "predicted_max_ft": 6.5,
    "predicted_min_ft": 2.1,
    "tide_range": 4.4,
    "tide_level": 3.5,
    "tugs": 1,
    "barges": 2,
    "tankers": 1,
    "total_vessels": 4,
    "weather_score": 7,
    "cold_day": 0,
    "heat_alert": 0,
    "snow_impact": 0,
    "rainy": 0,
    "has_tanker": 1,
    "has_barge": 1,
    "dir_IN": 1,
    "dir_OUT": 0,
    "dir_IN/OUT": 0
}

X_input = pd.DataFrame([features])
X_scaled = scaler.transform(X_input)
y_pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0]
confidence = round(np.max(proba) * 100, 1)
estimated_duration = random.randint(15, 20)

next_lift_times = [
    (now + timedelta(minutes=14 + i * 90)).astimezone(BOSTON_TZ).strftime("%I:%M %p %Z")
    for i in range(3)
]

# Confidence gauge setup
bar_color = "#4CAF50" if confidence >= 80 else "#FFB84D" if confidence >= 60 else "#FF4C4C"
bg_color = "#E8F5E9" if confidence >= 80 else "#FFF5E5" if confidence >= 60 else "#FFEAEA"

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=confidence,
    number={'suffix': "%", 'font': {'size': 30, 'color': "#333"}},
    gauge={
        'shape': "angular",
        'axis': {'range': [0, 100], 'visible': False},
        'bar': {'color': bar_color, 'thickness': 0.3},
        'bgcolor': bg_color,
        'borderwidth': 2,
        'bordercolor': "#eee",
        'threshold': {
            'line': {'color': bar_color, 'width': 3},
            'thickness': 0.8,
            'value': confidence
        }
    }
))
fig.update_layout(width=100, height=100, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")

# Header
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background-color: #fafbfc; padding: 20px 0; border-radius: 8px; margin-bottom: 20px;">
        <div style="flex: 1; display: flex; justify-content: flex-start;">
            <img src='https://img.masstransitmag.com/files/base/cygnus/mass/image/2014/09/massdot-logo_11678559.png' style='height: 80px;' />
        </div>
        <div style="flex: 2; text-align: center;">
            <span style="font-size: 34px; font-weight: 700; color: #003366;">Chelsea Bridge Lift Forecast System</span>
        </div>
        <div style="flex: 1; text-align: right; font-size: 16px; color: #333;">
            Last updated at: <strong>{now.strftime('%-I:%M %p %Z')}</strong>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["Home", "Notifications", "Historical Trends", "About"])

# ----------------- HOME TAB -----------------
with tabs[0]:
    col1, spacer, col2, col3 = st.columns([1, 0.2, 1, 1])

    with col1:
        st.markdown("<h3 style='color:#003366; text-align:center;'>Next Predicted</h3>", unsafe_allow_html=True)
        for t in next_lift_times:
            st.markdown(f"""
                <div style='background-color: white; color: #003366; font-size: 28px; 
                        font-weight: bold; padding: 10px 16px; border-radius: 12px; 
                        margin-bottom: 43px; box-shadow: 2px 2px 6px rgba(0,0,0,0.12); 
                        text-align: center;'>
                    {t}
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='color:#003366; text-align:center;'>Confidence Score</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig)

        st.markdown("<div style='height:36px;'></div>", unsafe_allow_html=True)
        bridge_status = "OPEN" if time_since_last_lift < 20 else "CLOSED"
        status_color = "red" if bridge_status == "OPEN" else "green"
        status_bg = "#fff2f2" if bridge_status == "OPEN" else "#e7f8ec"

        st.markdown(f"""
            <h3 style='color:#003366; text-align:center;'>Bridge Status</h3>
            <div style='background-color: {status_bg}; color: {status_color}; font-size: 26px; font-weight: bold; padding: 14px; border-radius: 12px; text-align: center; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);'>
                {bridge_status}
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("<h3 style='color:#003366; text-align:center;'>Estimated Duration</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='background-color: white; font-size: 24px; font-weight: bold; color: #003366; padding: 14px; text-align: center; border-radius: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);'>
                {estimated_duration} mins
            </div>
            <div style='background-color: #d9eaff; padding: 12px; border-radius: 12px; margin-top: 120px; font-size: 16px; text-align: center;'>
                <p style='font-size:18px; margin-bottom: 8px;'>🌤️ <strong>Weather:</strong> Clear, 66°F</p>
                <p>🌊 <strong>Tide Level:</strong> Mid Tide</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color:#f1f3f5; padding:12px; text-align:center; font-size:12px; color: #555;'>
            Powered by ChelseaBridgeAI | Historical data (2019–2025)
        </div>
    """, unsafe_allow_html=True)


with tabs[1]:
    col_vms, col_tweet = st.columns(2)

    # --- VMS Panel ---
    with col_vms:
        st.markdown(f"""
            <div style='
                background-color: #1a1a1a;
                color: #ff9900;
                font-family: monospace;
                font-size: 28px;
                padding: 24px;
                line-height: 1.6;
                border-radius: 12px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
                text-align: center;
                min-height: 400px;
                display: flex;
                flex-direction: column;
                align-items: center;
            '>
                <div style='
                    background:#006400;
                    color:#fff;
                    padding:6px 0;
                    border-radius:6px;
                    font-size:18px;
                    line-height:1.4;
                    width: 100%;
                    margin-bottom: 28px;
                '>
                    NEXT LIFT EXPECTED<br>
                    SIGUIENTE LEVADIZO ESPERADO
                </div>
                <div style='
                    flex: 1;
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-evenly;
                    align-items: center;
                    gap: 24px;
                '>
                    {"".join([f"<div style='font-size: 36px; font-weight: bold; color: orange; letter-spacing: 1px;'>{t}</div>" for t in next_lift_times])}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.button("📤 Publish to Signboard", key="vms_button")

    # --- Twitter Panel ---
    with col_tweet:
        # Boston-localized time window
        start_time = now.replace(hour=7, minute=0, second=0, microsecond=0)
        end_time = now.replace(hour=19, minute=30, second=0, microsecond=0)

        homepage_times = [
            datetime.strptime(t, "%I:%M %p %Z").replace(
                year=now.year, month=now.month, day=now.day,
                tzinfo=BOSTON_TZ
            )
            for t in next_lift_times
        ]

        day_lifts = [(t, random.choice([15, 20, 25])) for t in homepage_times]

        attempts = 0
        while len(day_lifts) < 10 and attempts < 150:
            candidate = start_time + timedelta(minutes=random.randint(0, 770))
            if start_time <= candidate <= end_time:
                if all(abs((candidate - existing[0]).total_seconds()) >= 90 * 60 for existing in day_lifts):
                    day_lifts.append((candidate, random.choice([15, 20, 25])))
            attempts += 1

        # Ensure first homepage time has actual homepage-estimated duration
        day_lifts = [(lt, estimated_duration if lt == homepage_times[0] else dur) for lt, dur in day_lifts]
        day_lifts = sorted(day_lifts, key=lambda x: x[0])

        # Format the lifts as readable strings
        lift_schedule_strs = [
            f"{lt.strftime('%I:%M %p')} — estimated duration {dur} mins"
            for lt, dur in day_lifts
        ]

        st.markdown(f"""
            <div style='
                border: 1px solid #ccc;
                border-radius: 12px;
                padding: 20px;
                background-color: #f9f9f9;
                font-family: Arial, sans-serif;
                font-size: 16px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                min-height: 400px;
            '>
                <strong style='font-size:18px;'>@LoganToChelsea</strong> &nbsp;
                <span style='color:#555;'>· {now.strftime('%b %d')}</span>
                <p style='margin-top:10px; margin-bottom:10px;'>
                    {now.month}/{now.day} Expected Bridge Lifts<br>
                    {"<br>".join(lift_schedule_strs)}<br><br>
                    <em>*Subject to change</em>
                </p>
                <span style='color:#1DA1F2;'>Twitter Web App</span>
            </div>
        """, unsafe_allow_html=True)

        st.button("Post Tweet Update", key="tweet_button")

    # Footer
    st.markdown("""
        <div style='background-color:#f1f3f5; padding:12px; text-align:center; font-size:12px; color: #555;'>
            Powered by ChelseaBridgeAI | Historical data (2019–2025)
        </div>
    """, unsafe_allow_html=True)


with tabs[2]:
    st.markdown("<h3 style='text-align: center;'>Historical Lift Trends (Weekly Averages)</h3>", unsafe_allow_html=True)

    # Load and process data
    df = pd.read_csv("Final_cleaned_data.csv")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["hour"] = df["start_time"].dt.hour
    df["day_name"] = df["start_time"].dt.day_name()

    df = df[df["hour"].between(7, 19)]

    ## 1. Heatmap: Weekly lift distribution
    heatmap_df = df.groupby(["day_name", "hour"]).size().reset_index(name="count")
    heatmap_df["day_name"] = pd.Categorical(
        heatmap_df["day_name"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True
    )
    heatmap_df = heatmap_df.sort_values(["day_name", "hour"])

    heatmap = px.density_heatmap(
        heatmap_df, x="hour", y="day_name", z="count",
        color_continuous_scale="Blues", nbinsx=13,
        labels={"hour": "Hour of Day", "day_name": "Day", "count": "Lift Count"},
        title="Peak Lift Times Across Week",
        category_orders={"day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
    )

    heatmap.update_layout(
        title={
            'text': "Peak Lift Times Across Week",
            'x': 0.5,
            'xanchor': 'center'
        },
        title_font=dict(size=22),
        font=dict(size=16),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )


    st.plotly_chart(heatmap, use_container_width=True)


    ## 2. Bar Chart: Vessel type frequency
    vessel_counts = df[["tugs", "barges", "tankers"]].sum().reset_index()
    vessel_counts.columns = ["Vessel", "Count"]
    bar_chart = px.bar(
        vessel_counts, x="Vessel", y="Count", text="Count",
        color="Vessel", color_discrete_sequence=["#1f77b4", "#4682b4", "#5dade2"],
        title="Vessel Type Frequencies Triggering Lifts"
    )
    bar_chart.update_traces(textposition="outside")
    bar_chart.update_layout(
        title={
            'text': "Vessel Type Frequencies Triggering Lifts",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    ## 3. Donut Chart: Tide level distribution
    tide_dist = df["tide_level"].apply(
        lambda x: "Low" if x < 3 else "Mid" if x < 6 else "High"
    ).value_counts().reset_index()
    tide_dist.columns = ["Tide Level", "Count"]
    donut = px.pie(
        tide_dist, names="Tide Level", values="Count", hole=0.5,
        color_discrete_sequence=["#87cefa", "#4682b4", "#1e90ff"],
        title="Distribution of Tide Levels During Lifts"
    )
    donut.update_layout(
        title={
            'text': "Distribution of Tide Levels During Lifts",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        showlegend=True
    )

    st.plotly_chart(donut, use_container_width=True)

    ## 4. Histogram: Lift duration
    df["duration_mins"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60

    hist = px.histogram(
        df,
        x="duration_mins",
        labels={"duration_mins": "Duration (minutes)"},
        color_discrete_sequence=["#4682b4"],
        range_x=[0, 30],
    )

    # Explicit binning: 6 bins → width of 5 minutes each
    hist.update_traces(xbins=dict(start=0, end=30, size=5))

    hist.update_layout(
        title={
            'text': "Lift Duration Ranges (in minutes)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=16)
    )

    st.plotly_chart(hist, use_container_width=True)

    st.markdown("""
        <div style='background-color:#f1f3f5; padding:12px; text-align:center; font-size:12px; color: #555;'>
            Powered by ChelseaBridgeAI | Historical data (2019–2025)
        </div>
    """, unsafe_allow_html=True)

with tabs[3]:
    st.markdown("""
            The <strong>Chelsea Bridge Lift Forecast System</strong> is an intelligent dashboard designed to provide accurate, real-time predictions 
            of bridge lift timings along with operational insights for stakeholders and commuters. Powered by machine learning, the system uses 
            historical data, vessel patterns, tidal conditions, and weather features to predict the likelihood and timing of upcoming bridge lifts.
        </p>

        <h4 style='margin-top: 30px; color: #003366;'>Key Features</h4>
        <ul style='font-size:15px; line-height:1.7;'>
            <li>Real-time forecast of next lift timings and estimated durations</li>
            <li>Visual analytics on historical trends, tide patterns, and vessel activity</li>
            <li>Publish updates via digital signage and social channels</li>
            <li>Data-driven insights to support maritime planning and traffic coordination</li>
        </ul>

        <h4 style='margin-top: 30px; color: #003366;'>Data & Model</h4>
        <p style='font-size:15px; text-align: justify;'>
            The system is trained on Chelsea Street Bridge data from 2019–2025, using a LightGBM classifier with carefully engineered features such as 
            time since last lift, number and type of vessels, tidal data, and weather metrics.
        </p>

        <h4 style='margin-top: 30px; color: #003366;'>Acknowledgements</h4>
        <p style='font-size:15px; text-align: justify;'>
            This project was developed as part of an analytics and machine learning initiative to modernize infrastructure intelligence systems.
            Special thanks to the teams supporting open data access, and to MassDOT for operational insights.
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background-color:#f1f3f5; padding:12px; text-align:center; font-size:12px; color: #555;'>
            Powered by ChelseaBridgeAI | Historical data (2019–2025)
        </div>
    """, unsafe_allow_html=True)
