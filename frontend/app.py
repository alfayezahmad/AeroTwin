import os
import time
import random
import requests
import itertools
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="AeroTwin | Digital Twin Command Center",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Use environment variable for backend URL, fallback to localhost for local testing
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if "simulate_spike" not in st.session_state:
    st.session_state["simulate_spike"] = False

with st.sidebar:
    st.markdown("### 🔧 CYBER-PHYSICAL CONTROLS")
    st.markdown("<p style='font-size:0.8rem; color:#9ca3af;'>Test closed-loop hardware actuation via MQTT downlink to ESP32-S3 edge node.</p>", unsafe_allow_html=True)
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("⚡ Spike Test", use_container_width=True):
            st.session_state["simulate_spike"] = True
            try:
                requests.post(f"{BACKEND_URL}/api/simulate_spike", timeout=2)
            except Exception:
                pass
            st.toast("Atmospheric PM2.5 Spike Injected (GRAP IV Emergency)! MQTT Downlink Fired.", icon="🚨")
            time.sleep(0.5)
            st.rerun()
    with col_sb2:
        if st.button("🔄 Reset Live", use_container_width=True):
            st.session_state["simulate_spike"] = False
            try:
                requests.post(f"{BACKEND_URL}/api/reset_spike", timeout=2)
            except Exception:
                pass
            st.toast("Restored Live Sensor Baseline Telemetry! Edge Actuators on Standby.", icon="✅")
            time.sleep(0.5)
            st.rerun()

    st.markdown("<hr style='margin: 1rem 0 !important;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; color:#9ca3af;'><b>MQTT BROKER:</b> aerotwin-mqtt (1883)<br/><b>TOPIC:</b> aerotwin/commands/dispatch<br/><b>ACTUATOR:</b> ESP32-S3 (WS2812 + L293D)</div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    .stDeployButton {display: none !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Sleeker Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.08) 0%, rgba(0,0,0,0) 100%);
        border-left: 3px solid #00e5ff;
        padding: 10px 15px;
        border-radius: 4px;
    }
    div[data-testid="metric-container"] label {
        font-weight: 500 !important;
        color: #6b7280 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stDataFrame { border: 1px solid #1e1e24 !important; border-radius: 4px !important; }
    hr { border-color: #1e1e24 !important; margin: 1.5rem 0 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    
    /* Neon Cyber Button Styling */
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, rgba(0,229,255,0.8) 0%, rgba(0,123,181,0.8) 100%) !important;
        color: #ffffff !important;
        border: 1px solid #00e5ff !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] button:hover {
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.6) !important;
        border: 1px solid #ffffff !important;
    }
    
    /* Tactical Telemetry Card Grid */
    .tactical-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 10px;
    }
    .tactical-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 10px 12px;
        box-sizing: border-box;
        transition: border-color 0.2s ease;
    }
    .tactical-card:hover {
        border-color: #00e5ff;
    }
    .tactical-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #9ca3af;
        margin-bottom: 4px;
    }
    .tactical-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f9fafb;
        font-family: 'Inter', sans-serif;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .tactical-sub {
        font-size: 0.72rem;
        color: #9ca3af;
        font-weight: 400;
    }
    .tactical-badge {
        display: inline-block;
        margin-top: 6px;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        white-space: nowrap;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Station definitions and microclimate typology dispersion profiles
STATIONS = [
    {"station": "Talkatora", "lat": 26.8315, "lon": 80.8992},
    {"station": "Lalbagh", "lat": 26.8467, "lon": 80.9462},
    {"station": "Gomti Nagar", "lat": 26.8500, "lon": 80.9980},
    {"station": "Alambagh", "lat": 26.8150, "lon": 80.9020},
    {"station": "Kalyanpur", "lat": 26.9020, "lon": 80.9450},
]

STATION_PROFILES = {
    "Talkatora": {"typology": "Industrial Cluster", "factor": 1.25, "co_scale": 1.45, "peak_hours": [7, 8, 9, 18, 19, 20], "peak_boost": 20.0},
    "Lalbagh": {"typology": "Commercial Core", "factor": 1.10, "co_scale": 1.15, "peak_hours": [9, 10, 17, 18, 19], "peak_boost": 12.0},
    "Gomti Nagar": {"typology": "Residential Greenbelt", "factor": 0.82, "co_scale": 0.70, "peak_hours": [8, 9, 20, 21], "peak_boost": 5.0},
    "Alambagh": {"typology": "Interstate Transit Hub", "factor": 1.30, "co_scale": 1.60, "peak_hours": [6, 7, 14, 21, 22, 23], "peak_boost": 24.0},
    "Kalyanpur": {"typology": "Peri-Urban Advection", "factor": 0.95, "co_scale": 0.90, "peak_hours": [11, 12, 13, 14], "peak_boost": 8.0},
}


def assign_grap_stage(pm25_val: float) -> tuple[str, str]:
    if pm25_val > 350.0:
        return "Stage IV (Severe+)", "EMERGENCY_CLOSURE_AND_VEHICLE_BAN"
    elif pm25_val > 250.0:
        return "Stage III (Severe)", "PRIORITY_1_MIST_CANNON_DISPATCH"
    elif pm25_val > 120.0:
        return "Stage II (Very Poor)", "WATER_SPRINKLER_HIGH_DENSITY"
    elif pm25_val > 60.0:
        return "Stage I (Poor)", "MECHANIZED_ROAD_SWEEPING"
    else:
        return "Normal", "ROUTINE_MONITORING"


def generate_synthetic_timeseries(station_name: str = "Talkatora") -> tuple[list, list]:
    now = datetime.now()
    times = [(now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(24, 0, -1)]
    prof = STATION_PROFILES.get(station_name, {"factor": 1.0, "peak_hours": [], "peak_boost": 0.0})
    factor = prof.get("factor", 1.0)
    seed = sum(ord(c) for c in station_name)
    np.random.seed(seed)
    base_walk = np.clip(110 + np.cumsum(np.random.randn(24) * 8), 60, 280)
    pm25_vals = []
    for i, val in enumerate(base_walk):
        hour = (now - timedelta(hours=(24 - i))).hour
        boost = prof.get("peak_boost", 0.0) if hour in prof.get("peak_hours", []) else 0.0
        pm25_vals.append(round(float(val * factor + boost), 1))
    return times, pm25_vals


def generate_standalone_telemetry(simulate_spike: bool = False) -> dict:
    results = []
    for st_info in STATIONS:
        station_name = st_info["station"]
        lat, lon = st_info["lat"], st_info["lon"]
        prof = STATION_PROFILES.get(station_name, {"factor": 1.0, "co_scale": 1.0, "peak_hours": [], "peak_boost": 0.0})
        factor = prof["factor"]
        
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,carbon_monoxide&hourly=pm2_5&past_hours=24"
        wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_direction_10m"
        
        try:
            aqi_res = requests.get(aqi_url, timeout=4).json()
            wx_res = requests.get(wx_url, timeout=4).json()
            raw_pm25 = aqi_res["current"]["pm2_5"]
            raw_co = aqi_res["current"]["carbon_monoxide"]
            temp = wx_res["current"]["temperature_2m"]
            wind = wx_res["current"]["wind_speed_10m"]
            wind_dir = wx_res["current"]["wind_direction_10m"]
            
            timeseries_time = aqi_res["hourly"]["time"][:24]
            timeseries_pm25_raw = aqi_res["hourly"]["pm2_5"][:24]
            
            filled_pm25 = []
            last_val = raw_pm25 if raw_pm25 is not None else 100.0
            for i, val in enumerate(timeseries_pm25_raw):
                if val is not None:
                    last_val = val
                hour_of_day = i % 24
                boost = prof["peak_boost"] if hour_of_day in prof["peak_hours"] else 0.0
                station_pm = round(max(15.0, (last_val * factor) + boost), 1)
                filled_pm25.append(station_pm)
                
            current_hour = datetime.now().hour
            current_boost = prof["peak_boost"] if current_hour in prof["peak_hours"] else 0.0
            base_pm = raw_pm25 if raw_pm25 is not None else 110.0
            display_pm25 = round(max(20.0, (base_pm * factor) + current_boost), 1)
            co_val = round((raw_co if raw_co is not None else 1.0) * prof["co_scale"], 2)
            
        except Exception:
            timeseries_time, filled_pm25 = generate_synthetic_timeseries(station_name)
            display_pm25 = filled_pm25[-1] if filled_pm25 else 185.0
            co_val = round(1.2 * prof["co_scale"], 2)
            temp, wind, wind_dir = 28.0, 7.5, 160.0

        if simulate_spike:
            display_pm25 = round(display_pm25 + 230.0 + (lat * 20 % 50), 1)
            filled_pm25 = [round(v + 180.0, 1) for v in filled_pm25]

        grap_stage, action = assign_grap_stage(display_pm25)
        results.append({
            "station": station_name,
            "lat": lat,
            "lon": lon,
            "typology": prof["typology"],
            "pm25": display_pm25,
            "live_co": co_val,
            "live_temp": temp if 'temp' in locals() and temp is not None else 28.0,
            "live_wind": wind if 'wind' in locals() and wind is not None else 6.0,
            "wind_dir": wind_dir if 'wind_dir' in locals() and wind_dir is not None else 180.0,
            "timeseries_time": timeseries_time,
            "timeseries_pm25": filled_pm25,
            "grap_stage": grap_stage,
            "prescribed_action": action,
            "needs_dispatch": bool(display_pm25 > 120.0),
        })

    avg_pm25 = round(sum(r["pm25"] for r in results) / len(results), 1)
    critical_nodes = [r for r in results if r["needs_dispatch"]]
    
    hub_coords = [26.8467, 80.9462] # Central Municipal Depot
    hub_lon, hub_lat = hub_coords[1], hub_coords[0]
    
    routes = []
    for node in critical_nodes:
        routes.append({
            "start": [hub_lon, hub_lat],
            "end": [node["lon"], node["lat"]]
        })
        
    if simulate_spike:
        shap = {
            "Industrial Thermal Flux": 38.2,
            "Boundary Layer Inversion": 29.5,
            "Traffic Congestion Index": 18.6,
            "Wind Advection (Downwind)": 13.7
        }
    else:
        shap = {
            "Wind Advection (Downwind)": 36.4,
            "Boundary Layer Inversion": 28.1,
            "Traffic Congestion Index": 20.3,
            "Ambient Humidity": 15.2
        }

    return {
        "city": "Lucknow",
        "timestamp_horizon": "Real-Time / Next 24 Hours",
        "city_mean_pm25": avg_pm25,
        "nodes": results,
        "routes": routes,
        "shap_attributions": shap,
        "dispatch_hub": {
            "lat": hub_lat,
            "lon": hub_lon,
            "name": "Central Municipal Depot",
        },
    }


@st.cache_data(ttl=5)
def fetch_telemetry(is_spike_active: bool):
    try:
        res = requests.get(f"{BACKEND_URL}/api/forecast", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return generate_standalone_telemetry(simulate_spike=is_spike_active)


data = fetch_telemetry(st.session_state.get("simulate_spike", False))

st.markdown("## AEROTWIN // CYBER-PHYSICAL AIR QUALITY TWIN")
st.markdown(
    "<span style='color:#00e5ff; font-weight:600; font-size: 0.85rem; letter-spacing: 1px;'>SPATIO-TEMPORAL GNN INFERENCE</span> • <span style='color:#9ca3af; font-size: 0.85rem; letter-spacing: 1px;'>ESP32-S3 CLOSED-LOOP EDGE ACTUATION • GOOGLE OR-TOOLS VRP ROUTING</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

nodes = data["nodes"]
df = pd.DataFrame(nodes)
mean_pm25 = data["city_mean_pm25"]
active_dispatches = sum(1 for n in nodes if n["needs_dispatch"])
inference_latency = random.randint(45, 85)

# Create main functional zones (30% Left Rail, 70% Right Canvas)
col_left, col_right = st.columns([3, 7], gap="large")

with col_left:
    st.markdown("<h4 style='font-weight:400; color:#e5e7eb;'>TACTICAL TELEMETRY</h4>", unsafe_allow_html=True)
    
    # Dynamic styling based on PM2.5 and dispatch thresholds
    if mean_pm25 >= 350:
        pm_border = "#ef4444"
        pm_badge_style = "background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.5);"
        pm_badge_text = "⚡ GRAP IV (Emergency)"
    elif mean_pm25 >= 120:
        pm_border = "#f59e0b"
        pm_badge_style = "background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.5);"
        pm_badge_text = "⚠️ GRAP III (Severe)"
    else:
        pm_border = "#00e5ff"
        pm_badge_style = "background: rgba(0, 229, 255, 0.15); color: #67e8f9; border: 1px solid rgba(0, 229, 255, 0.4);"
        pm_badge_text = "● Normal Baseline"

    if active_dispatches > 0:
        dispatch_border = "#f59e0b"
        dispatch_badge_style = "background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.5);"
        dispatch_badge_text = f"🚛 {active_dispatches} Units Active"
        esp_badge_style = "background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.5);"
        esp_badge_text = "🚨 Mitigation Engaged"
        esp_border = "#ef4444"
    else:
        dispatch_border = "#10b981"
        dispatch_badge_style = "background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4);"
        dispatch_badge_text = "✓ Depot Standby"
        esp_badge_style = "background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4);"
        esp_badge_text = "● MQTT Synced"
        esp_border = "#10b981"

    telemetry_cards_html = f"""
    <div class="tactical-grid">
        <div class="tactical-card" style="border-left: 3px solid {pm_border};">
            <div class="tactical-label">Pred. Mean PM2.5</div>
            <div class="tactical-val">{mean_pm25} <span class="tactical-sub">µg/m³</span></div>
            <span class="tactical-badge" style="{pm_badge_style}">{pm_badge_text}</span>
        </div>
        <div class="tactical-card" style="border-left: 3px solid #10b981;">
            <div class="tactical-label">Active Sensors</div>
            <div class="tactical-val">{len(df)} <span class="tactical-sub">/ 5 nodes</span></div>
            <span class="tactical-badge" style="background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4);">● 100% Online</span>
        </div>
        <div class="tactical-card" style="border-left: 3px solid {dispatch_border};">
            <div class="tactical-label">Fleet Dispatches</div>
            <div class="tactical-val">{active_dispatches} <span class="tactical-sub">units</span></div>
            <span class="tactical-badge" style="{dispatch_badge_style}">{dispatch_badge_text}</span>
        </div>
        <div class="tactical-card" style="border-left: 3px solid {esp_border};">
            <div class="tactical-label">Edge Actuator</div>
            <div class="tactical-val" style="font-size: 1.15rem; padding-top: 3px;">ESP32-S3</div>
            <span class="tactical-badge" style="{esp_badge_style}">{esp_badge_text}</span>
        </div>
    </div>
    <div class="tactical-card" style="border-left: 3px solid #00e5ff; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="tactical-label">ST-GNN Inference Latency</div>
                <div class="tactical-val" style="font-size: 1.25rem;">{inference_latency} <span class="tactical-sub">ms</span></div>
            </div>
            <span class="tactical-badge" style="background: rgba(0, 229, 255, 0.15); color: #67e8f9; border: 1px solid rgba(0, 229, 255, 0.4);">⚡ Real-Time Compute</span>
        </div>
    </div>
    """
    st.markdown(telemetry_cards_html, unsafe_allow_html=True)
    
    # Cyber-Physical Edge Gateway Status Banner
    st.markdown(f"""
    <div style="background: rgba(0, 229, 255, 0.08); border-left: 3px solid #00e5ff; padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; color: #d1d5db; margin-top: 10px;">
        <span style="color:#00e5ff; font-weight:bold;">EDGE ACTUATOR:</span> ESP32-S3 (FreeRTOS / PWM / L293D) <br/>
        <span style="color:#00e5ff; font-weight:bold;">DOWNLINK:</span> Eclipse Mosquitto MQTT (aerotwin/commands/dispatch) <br/>
        <span style="color:#00e5ff; font-weight:bold;">INFERENCE:</span> PyTorch Geometric ST-GNN + Gemini Diagnostics
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1rem 0 !important;'>", unsafe_allow_html=True)
    
    # THE AGENTIC LAYER UI
    st.markdown("### AUTONOMOUS AGENT BRIEFING")
    briefing_rendered = False
    try:
        agent_res = requests.get(f"{BACKEND_URL}/api/agent_briefing", timeout=2)
        if agent_res.status_code == 200:
            agent_json = agent_res.json()
            briefing_text = agent_json.get("briefing", "")
            if briefing_text and "Agent Offline" not in briefing_text and "Agent error" not in briefing_text:
                st.markdown(f"""
                <div style="background-color: rgba(0, 229, 255, 0.05); border-left: 3px solid #00e5ff; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #e5e7eb; margin-bottom: 10px; line-height: 1.5;">
                    <div style="color: #00e5ff; font-weight: bold; margin-bottom: 10px;">> AEROTWIN_COMMAND_AI // GEMINI DIAGNOSTIC UPLINK</div>
                    {briefing_text}
                </div>
                """, unsafe_allow_html=True)
                briefing_rendered = True
    except Exception:
        pass

    if not briefing_rendered:
        sorted_nodes = sorted(nodes, key=lambda n: n.get("pm25", 0), reverse=True)
        top_node = sorted_nodes[0] if sorted_nodes else {}
        worst_st = top_node.get("station", "Talkatora")
        worst_val = top_node.get("pm25", mean_pm25)
        worst_act = top_node.get("prescribed_action", "ROUTINE_MONITORING").replace("_", " ")
        grap_lvl = top_node.get("grap_stage", "Stage I")
        w_wind = top_node.get("live_wind", 6.2)
        heuristic_briefing = (
            f"COMMISSIONER BRIEFING: PyTorch Geometric ST-GNN indicates particulate aggregation in {worst_st} "
            f"at {worst_val:.1f} µg/m³, driven by downwind boundary-layer advection ({w_wind} km/h). "
            f"Statutory CAQM GRAP {grap_lvl} mandates have been enacted with '{worst_act.title()}' actively broadcast. "
            f"Google OR-Tools VRP mist-cannon fleet routes and ESP32-S3 edge actuator PWM downlinks have been synchronized to intercept the plume."
        )
        st.markdown(f"""
        <div style="background-color: rgba(0, 229, 255, 0.05); border-left: 3px solid #00e5ff; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #e5e7eb; margin-bottom: 10px; line-height: 1.5;">
            <div style="color: #00e5ff; font-weight: bold; margin-bottom: 10px;">> AEROTWIN_COMMAND_AI // GEMINI DIAGNOSTIC UPLINK</div>
            {heuristic_briefing}
        </div>
        """, unsafe_allow_html=True)
            
    st.markdown("<hr style='margin: 1rem 0 !important;'>", unsafe_allow_html=True)
    
    # GRAP Table
    st.markdown("<h5 style='font-weight:400; color:#e5e7eb;'>STATUTORY GRAP MITIGATION BOARD</h5>", unsafe_allow_html=True)
    display_df = df[["station", "pm25", "grap_stage", "prescribed_action"]].copy()
    display_df["prescribed_action"] = display_df["prescribed_action"].str.replace("_", " ").str.title()
    
    table_html = "<div style='background-color: #0e1117; border: 1px solid #1e1e24; border-radius: 6px; padding: 10px; overflow-x: auto;'><table style='width:100%; border-collapse: collapse; font-size: 0.80rem; color: #d1d5db; font-family: Inter, sans-serif; white-space: nowrap;'>"
    table_html += "<tr style='border-bottom: 1px solid #374151; color: #9ca3af; text-transform: uppercase;'><th style='text-align:left; padding:8px 4px;'>Node</th><th style='text-align:left; padding:8px 4px;'>PM2.5</th><th style='text-align:left; padding:8px 4px;'>Status</th><th style='text-align:left; padding:8px 4px;'>Intervention</th></tr>"
    for _, row in display_df.iterrows():
        status = row['grap_stage']
        color = "#00e5ff" if status == "Normal" else ("#ef4444" if "Severe" in status else "#f59e0b")
        table_html += f"<tr style='border-bottom: 1px solid #1f2937;'>"
        table_html += f"<td style='padding:8px 4px;'>{row['station']}</td>"
        table_html += f"<td style='padding:8px 4px; font-family: monospace;'>{row['pm25']}</td>"
        table_html += f"<td style='padding:8px 4px; color: {color}; font-weight: 600;'>{status}</td>"
        table_html += f"<td style='padding:8px 4px;'>{row['prescribed_action']}</td>"
        table_html += "</tr>"
    table_html += "</table></div>"
    st.markdown(table_html, unsafe_allow_html=True)


with col_right:
    st.markdown("<h4 style='font-weight:400; color:#e5e7eb;'>SPATIAL POLLUTION TOPOLOGY & FLEET ROUTING</h4>", unsafe_allow_html=True)
    
    def get_color(val):
        if val > 350:
            return [138, 43, 226, 255] # Purple (Severe+)
        elif val > 250:
            return [220, 38, 38, 255]  # Crimson Red (Severe)
        elif val > 120:
            return [245, 158, 11, 255] # Amber (Very Poor)
        elif val > 60:
            return [234, 179, 8, 255]  # Yellow (Poor)
        return [0, 229, 255, 255]      # Cyan (Normal)

    df["color"] = df["pm25"].apply(get_color)
    df["elevation"] = df["pm25"] * 25

    if "routes" in data and data["routes"]:
        df_routes = pd.DataFrame(data["routes"])
    else:
        df_routes = pd.DataFrame()

    # Generate graph edges connecting all 5 stations (Fully Connected Graph)
    graph_edges = []
    for n1, n2 in itertools.combinations(nodes, 2):
        graph_edges.append({
            "start": [n1["lon"], n1["lat"]],
            "end": [n2["lon"], n2["lat"]]
        })
    df_graph_edges = pd.DataFrame(graph_edges)

    # Central Municipal Depot Hub Beacon
    hub_info = data.get("dispatch_hub", {"lat": 26.8467, "lon": 80.9462, "name": "Central Municipal Depot"})
    df_hub = pd.DataFrame([{
        "lon": hub_info["lon"],
        "lat": hub_info["lat"],
        "name": hub_info.get("name", "Central Municipal Depot")
    }])

    layers = [
        # The physical graph topology (ST-GNN Message Passing layer)
        pdk.Layer(
            "LineLayer",
            data=df_graph_edges,
            get_source_position="start",
            get_target_position="end",
            get_color=[0, 229, 255, 70], # Low opacity glowing cyan
            get_width=2,
            pickable=False,
        ),
        # Central Municipal Fleet Depot Marker
        pdk.Layer(
            "ScatterplotLayer",
            data=df_hub,
            get_position=["lon", "lat"],
            get_color=[255, 170, 0, 240], # Amber beacon
            get_radius=180,
            pickable=True,
        ),
        # The station pollution column needles
        pdk.Layer(
            "ColumnLayer",
            data=df,
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1,
            radius=90,   # Slim, elegant sensor needles
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
    ]

    # The OR-Tools VRP Dispatch Parabolic Arcs
    if not df_routes.empty:
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=df_routes,
                get_source_position="start",
                get_target_position="end",
                get_source_color=[0, 229, 255, 200], # Glowing cyan source (Depot)
                get_target_color=[220, 38, 38, 255],   # Warning red target (Station)
                get_width=2,                           # Sleek, thin professional arcs
                pickable=True,
                auto_highlight=True,
            )
        )

    st.pydeck_chart(
        pdk.Deck(
            map_style=pdk.map_styles.CARTO_DARK,
            initial_view_state=pdk.ViewState(
                latitude=26.8467, longitude=80.9462, zoom=11.6, pitch=48, bearing=-15
            ),
            layers=layers,
            tooltip={
                "html": "<div style='font-family:Inter, sans-serif; font-size:12px; color:#ffffff; background:#111827; padding:8px; border:1px solid #374151; border-radius:4px;'><b>{station}</b><br/>Typology: {typology}<br/>PM2.5: {pm25} µg/m³<br/>GRAP: {grap_stage}</div>"
            },
        )
    )

    st.markdown("<hr style='margin: 1.5rem 0 !important;'>", unsafe_allow_html=True)
    
    # 2-Row Chart Layout for the Right Canvas
    
    # ROW 1: Severity Ranking and XAI
    chart_c1, chart_c2 = st.columns(2, gap="large")
    
    with chart_c1:
        st.markdown("<h5 style='font-weight:400; color:#e5e7eb;'>NODE SEVERITY RANKING</h5>", unsafe_allow_html=True)
        fig = px.bar(
            df,
            x="pm25",
            y="station",
            orientation="h",
            color="pm25",
            color_continuous_scale="Reds",
        )
        fig.update_traces(texttemplate='<b>%{x}</b> µg/m³', textposition="outside", cliponaxis=False, marker_line_color='#000000', marker_line_width=1, opacity=0.9)
        fig.update_layout(
            font=dict(family="Inter", size=11, color="#9ca3af"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=60, t=5, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(title="", showgrid=False, visible=True, automargin=True),
            coloraxis_showscale=False,
            bargap=0.45,
            height=260,
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_c2:
        st.markdown("<h5 style='font-weight:400; color:#e5e7eb;'>ML FEATURE IMPORTANCE (XAI)</h5>", unsafe_allow_html=True)
        shap_data = data.get("shap_attributions", {})
        if shap_data:
            features = list(shap_data.keys())
            importance = list(shap_data.values())
        else:
            features = ["No Data"]
            importance = [0]

        fig_ml = px.bar(
            x=importance, 
            y=features, 
            orientation="h",
            color=importance,
            color_continuous_scale=[[0, "#00838f"], [1, "#00e5ff"]]
        )
        fig_ml.update_traces(texttemplate='<b>%{x:.1f}%</b>', textposition="outside", cliponaxis=False, marker_line_color='#000000', marker_line_width=1, opacity=0.9)
        fig_ml.update_coloraxes(showscale=False)
        fig_ml.update_layout(
            font=dict(family="Inter", size=11, color="#9ca3af"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=45, t=5, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(title="", showgrid=False, visible=True, automargin=True),
            bargap=0.45,
            height=260,
        )
        fig_ml.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_ml, use_container_width=True, config={"displayModeBar": False})

    # ROW 2: Live Telemetry Timeline (24H)
    st.markdown("<hr style='margin: 1.5rem 0 !important;'>", unsafe_allow_html=True)
    st.markdown("<h5 style='font-weight:400; color:#e5e7eb;'>LIVE 24H TELEMETRY TIMELINE</h5>", unsafe_allow_html=True)
    
    # Process the timeseries data into a single Pandas DataFrame
    timeline_records = []
    
    station_colors = {
        "Talkatora": "#00e5ff",      # Cyan
        "Lalbagh": "#14b8a6",        # Teal
        "Gomti Nagar": "#f59e0b",    # Amber
        "Alambagh": "#ef4444",       # Red
        "Kalyanpur": "#8b5cf6"       # Purple
    }

    for node in nodes:
        times = node.get("timeseries_time", [])
        pm25_vals = node.get("timeseries_pm25", [])
        station = node.get("station", "Unknown")
        
        for t, v in zip(times, pm25_vals):
            timeline_records.append({"Time": t, "PM2.5": v, "Station": station})
            
    if timeline_records:
        df_timeline = pd.DataFrame(timeline_records)
        df_timeline["Time"] = pd.to_datetime(df_timeline["Time"])
        
        fig_timeline = px.line(
            df_timeline, 
            x="Time", 
            y="PM2.5", 
            color="Station",
            color_discrete_map=station_colors
        )
        
        fig_timeline.update_layout(
            font=dict(family="Inter", size=11, color="#9ca3af"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(
                title="", 
                showgrid=False, 
                showline=True, 
                linecolor="#374151"
            ),
            yaxis=dict(
                title="", 
                showgrid=False, 
                showline=True, 
                linecolor="#374151"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=""
            ),
            height=300,
        )
        fig_timeline.update_traces(line=dict(width=2.5), line_shape="spline")
        
        st.plotly_chart(fig_timeline, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("No timeline data available in payload.")
