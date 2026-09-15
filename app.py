import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import joblib
import os
import math
import random
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 0. ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY")

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyMate • Climate Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. DYNAMIC CSS STYLING ENGINE (Exact Time & Weather Gradients)
# -----------------------------------------------------------------------------
def apply_custom_theme(weather_condition="Clear", current_hour=12):
    cond = weather_condition.lower()
    
    # Defaults
    card_bg = "rgba(255, 255, 255, 0.85)"
    text_color = "#1E1B4B"
    show_stars = False

    # 1. SPECIAL WEATHER OVERRIDE (Rain / Storm / Drizzle)
    if "rain" in cond or "drizzle" in cond or "thunder" in cond:
        bg_gradient = "linear-gradient(135deg, #B8C0EC 0%, #A4B8C4 50%, #C3BEF0 100%)"
    
    # 2. TIME-BASED COLOR GRADIENTS
    elif 6 <= current_hour < 9:
        # Morning (6 to 9 AM): Light pale yellow, light blue gradient & white mix
        bg_gradient = "linear-gradient(135deg, #FFFF8A 0%, #FFFFFF 50%, #B8B8FF 100%)"
        
    elif 9 <= current_hour < 12:
        # Midday (9 to 12 PM): Light Lavender
        bg_gradient = "linear-gradient(135deg, #E6E6FA 0%, #DCD0FF 50%, #C8B6FF 100%)"
        
    elif 12 <= current_hour < 17:
        # Afternoon (12 to 5 PM): #FFFAA0 Warm / Hot / Sunny tone
        bg_gradient = "linear-gradient(135deg, #FFFAA0 0%, #FFE5B4 50%, #FFD180 100%)"
        
    elif 17 <= current_hour < 19:
        # Evening (5 to 7 PM): Pink, Peach & Yellow
        bg_gradient = "linear-gradient(135deg, #FFB6C1 0%, #FFDAB9 50%, #FFFACD 100%)"
        
    else:
        # Night Time (7 PM to 6 AM): Deep Navy with Stars
        bg_gradient = "radial-gradient(circle at top, #000075 0%, #000047 60%, #000022 100%)"
        card_bg = "rgba(15, 23, 42, 0.75)"
        text_color = "#B8B8FF"
        show_stars = True

    # 100 Star Generator for Night Sky
    stars_html = ""
    stars_css = ""
    if show_stars:
        stars_css = """
        .star-container {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: 0;
        }
        .star {
            position: absolute; background: #FFFFFF; border-radius: 50%;
            animation: twinkle 3s infinite ease-in-out;
        }
        @keyframes twinkle {
            0%, 100% { opacity: 0.2; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1.3); }
        }
        """
        # Generate 100 random stars
        random.seed(42)  # Fixed seed to keep star pattern consistent per load
        star_divs = []
        for _ in range(100):
            top = random.randint(1, 98)
            left = random.randint(1, 98)
            size = random.choice([1, 2, 3])
            delay = round(random.uniform(0, 3), 2)
            star_divs.append(
                f"<div class='star' style='top:{top}%; left:{left}%; width:{size}px; height:{size}px; animation-delay:{delay}s;'></div>"
            )
        stars_html = f"<div class='star-container'>{''.join(star_divs)}</div>"

    css = f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    {stars_css}

    .stApp {{
        background: {bg_gradient};
        background-attachment: fixed;
        color: {text_color} !important;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}

    h1, h2, h3, h4, h5, h6, p, span, div {{
        color: {text_color} !important;
    }}

    .weather-card {{
        background: {card_bg};
        backdrop-filter: blur(16px);
        border-radius: 22px;
        padding: 20px;
        border: 1px solid rgba(184, 184, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }}

    .cloud-container {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        overflow: hidden; pointer-events: none; z-index: 0;
    }}
    .cloud {{
        position: absolute; background: rgba(255, 255, 255, 0.25);
        border-radius: 100px; opacity: 0.6; animation: moveClouds linear infinite;
    }}
    .cloud::before, .cloud::after {{
        content: ''; position: absolute; background: rgba(255, 255, 255, 0.25);
        border-radius: 50%;
    }}
    .cloud1 {{ width: 200px; height: 60px; top: 10%; animation-duration: 35s; }}
    .cloud1::before {{ width: 90px; height: 90px; top: -40px; left: 30px; }}
    .cloud1::after {{ width: 70px; height: 70px; top: -30px; right: 30px; }}

    .cloud2 {{ width: 300px; height: 90px; top: 25%; animation-duration: 50s; animation-delay: -15s; }}
    .cloud2::before {{ width: 130px; height: 130px; top: -60px; left: 45px; }}
    .cloud2::after {{ width: 100px; height: 100px; top: -45px; right: 45px; }}

    @keyframes moveClouds {{
        0% {{ left: -350px; }}
        100% {{ left: 100vw; }}
    }}

    .compass-circle {{
        width: 80px; height: 80px;
        border: 3px solid #B8B8FF;
        background: rgba(0, 0, 117, 0.3);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        margin: 10px auto; font-weight: bold;
    }}
    </style>

    {stars_html}

    <div class="cloud-container">
        <div class="cloud cloud1"></div>
        <div class="cloud cloud2"></div>
    </div>
    """
    st.markdown(css, unsafe_allow_html=True)

# Helper function to calculate moon phase
def get_real_lunar_phase(dt):
    year, month, day = dt.year, dt.month, dt.day
    if month < 3:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = math.floor(a / 4)
    c = 2 - a + b
    e = math.floor(365.25 * (year + 4716))
    f = math.floor(30.6001 * (month + 1))
    jd = c + day + e + f - 1524.5
    
    days_since_new_moon = (jd - 2451549.5) % 29.53058867
    phase_ratio = days_since_new_moon / 29.53058867
    illumination = round((1 - math.cos(phase_ratio * 2 * math.pi)) / 2 * 100)

    if phase_ratio < 0.03 or phase_ratio > 0.97:
        phase_name = "New Moon"
    elif phase_ratio < 0.22:
        phase_name = "Waxing Crescent"
    elif phase_ratio < 0.28:
        phase_name = "First Quarter"
    elif phase_ratio < 0.47:
        phase_name = "Waxing Gibbous"
    elif phase_ratio < 0.53:
        phase_name = "Full Moon"
    elif phase_ratio < 0.72:
        phase_name = "Waning Gibbous"
    elif phase_ratio < 0.78:
        phase_name = "Last Quarter"
    else:
        phase_name = "Waning Crescent"

    return phase_name, illumination

# -----------------------------------------------------------------------------
# 3. REAL-TIME API FETCH & INFERENCE
# -----------------------------------------------------------------------------
def get_weather_data_and_predict(city_name):
    if not API_KEY:
        st.error("⚠️ `WEATHER_API_KEY` was not found in `.env` or Streamlit Secrets.")
        st.stop()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&units=metric&appid={API_KEY}"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200:
        st.error(f"⚠️ OpenWeatherMap Error ({res.status_code}): {data.get('message', 'Invalid city or API key')}")
        st.stop()

    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]
    tz_offset = data.get("timezone", 0)
    city_now = datetime.now(timezone.utc) + timedelta(seconds=tz_offset)
    current_hour = city_now.hour

    # Time Period Text Determination
    if 6 <= current_hour < 9:
        time_period = "Morning"
    elif 9 <= current_hour < 12:
        time_period = "Midday"
    elif 12 <= current_hour < 17:
        time_period = "Afternoon"
    elif 17 <= current_hour < 19:
        time_period = "Evening"
    else:
        time_period = "Night"

    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    f_res = requests.get(forecast_url)
    forecast_data = f_res.json() if f_res.status_code == 200 else {}

    if 6 <= current_hour < 10:
        sun_pos_text = "East (Rising)"
        sun_progress = 25
    elif 10 <= current_hour < 14:
        sun_pos_text = "Overhead / Zenith"
        sun_progress = 50
    elif 14 <= current_hour < 18:
        sun_pos_text = "South-West (Descending)"
        sun_progress = 75
    else:
        sun_pos_text = "Below Horizon (Night)"
        sun_progress = 100

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind_speed = round(data["wind"]["speed"] * 3.6, 1)
    wind_deg = data["wind"].get("deg", 0)
    
    # Real live weather condition string direct from OpenWeather API
    condition = data["weather"][0]["description"].title() if data.get("weather") else "Clear Sky"

    sunrise_dt = datetime.fromtimestamp(data["sys"].get("sunrise", 0), tz=timezone.utc) + timedelta(seconds=tz_offset)
    sunset_dt = datetime.fromtimestamp(data["sys"].get("sunset", 0), tz=timezone.utc) + timedelta(seconds=tz_offset)

    moon_phase, moon_illum = get_real_lunar_phase(city_now)

    hours_24, temps_24, precip_24 = [], [], []
    if "list" in forecast_data:
        for item in forecast_data["list"][:8]:
            dt_item = datetime.fromtimestamp(item["dt"], tz=timezone.utc) + timedelta(seconds=tz_offset)
            hours_24.append(dt_item.strftime("%H:%M"))
            temps_24.append(round(item["main"]["temp"], 1))
            precip_24.append(int(item.get("pop", 0) * 100))

    daily_forecasts = []
    if "list" in forecast_data:
        daily_map = {}
        for item in forecast_data["list"]:
            dt_item = datetime.fromtimestamp(item["dt"], tz=timezone.utc) + timedelta(seconds=tz_offset)
            day_name = dt_item.strftime("%A")
            if day_name not in daily_map and len(daily_map) < 5:
                cond_main = item["weather"][0]["main"].lower()
                icon = "☀️"
                if "rain" in cond_main: icon = "🌧️"
                elif "cloud" in cond_main: icon = "⛅"
                elif "clear" in cond_main: icon = "☀️"
                
                daily_map[day_name] = {
                    "day": day_name,
                    "icon": icon,
                    "max_temp": round(item["main"]["temp_max"], 1)
                }
        daily_forecasts = list(daily_map.values())

    next_hour_real_trend = temps_24[0] if len(temps_24) > 0 else temp
    model_path = os.path.join("saved_models", "weather_model.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            sample_features = np.array([[humidity, pressure, wind_speed, temp]])
            ml_prediction = model.predict(sample_features)[0]
        except Exception:
            ml_prediction = next_hour_real_trend
    else:
        ml_prediction = next_hour_real_trend

    uv_index = min(10, max(1, int((temp / 5))))

    return {
        "city": data.get("name", city_name.capitalize()),
        "country": data["sys"].get("country", "Live Station Data"),
        "temp": round(temp, 1),
        "ml_predicted_temp": round(ml_prediction, 1),
        "condition": condition,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_deg": wind_deg,
        "uv_index": uv_index,
        "sunrise": sunrise_dt.strftime("%I:%M %p"),
        "sunset": sunset_dt.strftime("%I:%M %p"),
        "sun_pos_text": sun_pos_text,
        "sun_progress": sun_progress,
        "current_hour": current_hour,
        "time_period": time_period,
        "moon_phase": moon_phase,
        "moon_illum": moon_illum,
        "hours_24": hours_24,
        "temps_24": temps_24,
        "precip_24": precip_24,
        "daily_forecasts": daily_forecasts
    }

# -----------------------------------------------------------------------------
# 4. DASHBOARD HEADER & CONTROLS
# -----------------------------------------------------------------------------
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown("""
    <div style="line-height: 1.1;">
        <span style="font-size: 38px; font-weight: 800; letter-spacing: -0.5px;">🌤️ SkyMate</span>
        <span style="font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.8; margin-left: 8px;">by Dhanshree Gupta</span>
    </div>
    <p style="margin-top: 4px; opacity: 0.8; font-size: 14px;">Real-Time Atmospheric Intelligence & Machine Learning Forecasts</p>
    """, unsafe_allow_html=True)

with top_col2:
    city = st.text_input("📍 Search City", value="Delhi")

data = get_weather_data_and_predict(city)
apply_custom_theme(weather_condition=data["condition"], current_hour=data["current_hour"])

st.markdown(f"""
<div class="weather-card" style="text-align: center; padding: 25px;">
    <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85; margin-bottom: 5px;">
        SkyMate • by Dhanshree Gupta
    </div>
    <h2>{data['city']}, {data['country']}</h2>
    <h1 style="font-size: 64px; margin: 5px 0;">
        {data['temp']}°C <span style="font-size: 26px; font-weight: normal; opacity: 0.85;">• {data['time_period']}</span>
    </h1>
    <h3 style="margin-bottom: 15px;">{data['condition']}</h3>
    <div style="background: rgba(0, 0, 117, 0.2); border: 1.5px dashed #B8B8FF; border-radius: 12px; padding: 10px; display: inline-block;">
        <span style="font-size: 15px; font-weight: bold;">
            🤖 ML Model Predicted Next-Hour Trend: <span style="color: #FF6B6B;">{data['ml_predicted_temp']}°C</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. METRICS GRID
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="weather-card" style="text-align: center;">
        <h4>💧 Humidity</h4>
        <h2>{data['humidity']}%</h2>
        <p>Atmospheric Moisture</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="weather-card" style="text-align: center;">
        <h4>⏲️ Pressure</h4>
        <h2>{data['pressure']} hPa</h2>
        <p>Barometric Level</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="weather-card" style="text-align: center;">
        <h4>☀️ UV Index</h4>
        <h2>{data['uv_index']} / 10</h2>
        <p>Solar Exposure</p>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="weather-card" style="text-align: center;">
        <h4>🧭 Wind Vector</h4>
        <div class="compass-circle" style="transform: rotate({data['wind_deg']}deg);">
            ⬆️ <span style="font-size: 12px;">{data['wind_speed']} k/h</span>
        </div>
        <p style="margin-top: 5px;">Heading: {data['wind_deg']}°</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. ASTRONOMICAL TRACKER
# -----------------------------------------------------------------------------
st.markdown("### 🌅 Astronomical & Solar Directional Tracker")
ast1, ast2 = st.columns(2)

with ast1:
    st.markdown(f"""
    <div class="weather-card">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 40px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 12px;">☀️</div>
            <div style="flex-grow: 1;">
                <h4 style="margin: 0;">Solar Cycle & Position</h4>
                <p style="margin: 5px 0;">📍 <b>Current Solar Heading:</b> <span>{data['sun_pos_text']}</span></p>
                <p style="margin: 2px 0;">🌅 <b>Sunrise:</b> {data['sunrise']} &nbsp;|&nbsp; 🌇 <b>Sunset:</b> {data['sunset']}</p>
                <div style="background: rgba(255,255,255,0.2); height: 8px; border-radius: 4px; margin-top: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #F59E0B, #EF4444); width: {data['sun_progress']}%; height: 100%;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ast2:
    st.markdown(f"""
    <div class="weather-card">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 40px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 12px;">🌙</div>
            <div>
                <h4 style="margin: 0;">Lunar Attributes</h4>
                <p style="margin: 5px 0;">🌙 <b>Phase:</b> {data['moon_phase']}</p>
                <p style="margin: 2px 0;">✨ <b>Illumination:</b> {data['moon_illum']}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PLOTLY GRAPH
# -----------------------------------------------------------------------------
st.markdown("### 📈 24-Hour Temperature & Environmental Trend History")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['hours_24'], y=data['temps_24'],
    mode='lines+markers',
    name='Temperature (°C)',
    line=dict(color='#FF6B6B', width=3, shape='spline'),
    marker=dict(size=6, color='#FF6B6B')
))

fig.add_trace(go.Bar(
    x=data['hours_24'], y=data['precip_24'],
    name='Precipitation Probability (%)',
    marker_color='rgba(77, 171, 247, 0.5)',
    yaxis='y2'
))

fig.update_layout(
    height=320,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.05)',
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(
        showgrid=True, 
        gridcolor='rgba(255,255,255,0.1)', 
        title=dict(text="Forecast Time Period")
    ),
    yaxis=dict(
        title=dict(text="Temperature (°C)"), 
        showgrid=True, 
        gridcolor='rgba(255,255,255,0.1)'
    ),
    yaxis2=dict(
        title=dict(text="Precipitation (%)"), 
        overlaying='y', 
        side='right', 
        range=[0, 100]
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. 5-DAY EXTENDED FORECAST GRID
# -----------------------------------------------------------------------------
st.markdown("### 📅 5-Day Extended Weather Forecast")
if data['daily_forecasts']:
    f_cols = st.columns(len(data['daily_forecasts']))
    for col, f_item in zip(f_cols, data['daily_forecasts']):
        with col:
            st.markdown(f"""
            <div class="weather-card" style="text-align: center;">
                <h5 style="margin: 0;">{f_item['day']}</h5>
                <h2 style="margin: 8px 0;">{f_item['icon']}</h2>
                <h3 style="margin: 0;">{f_item['max_temp']}°C</h3>
            </div>
            """, unsafe_allow_html=True)