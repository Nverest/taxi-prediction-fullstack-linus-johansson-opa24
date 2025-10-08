import streamlit as st
import datetime
from dotenv import load_dotenv
import requests
from streamlit_searchbox import st_searchbox
import folium
from streamlit_folium import st_folium
from taxipred.frontend.dashboard_util import (
    set_bg,
    geocode_address,
    get_time_of_day,
    fetch_address_suggestions,
    get_route_info,
)

load_dotenv()

# Page config
st.set_page_config(page_title="Taxi Price Prediction", layout="wide")
set_bg("visual_asset/taxi.png")

st.markdown("""
<style>

h1 {
    font-weight: 700;
    color: #FFD700 !important;
    font-size: 3.5rem !important;
    margin-bottom: 2rem !important;
    letter-spacing: -1px;
    /* Black border/stroke */
    -webkit-text-stroke: 2px black;
    text-stroke: 2px black;
    text-shadow: 0 4px 8px #FFA500;
}
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background: rgba(255, 215, 0, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 6px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border: 1px solid rgba(255, 215, 0, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1) !important;
        background: rgba(255, 255, 255, 0.15) !important;
    }

    /* Labels */
    .stTextInput > label,
    .stSelectbox > label,
    .stNumberInput > label,
    .stDateInput > label,
    .stTimeInput > label {
        color: #FFD700 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 10px !important;
    }

    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #1a1a1a !important;
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(255, 165, 0, 0.4) !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 16px 32px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255, 165, 0, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* Info cards styling */
    .info-card {
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.5) , transparent);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.3), inset 0 0 20px rgba(34, 197, 94, 0.1);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        margin-bottom: 24px;
        animation: pulse 2s ease-in-out infinite;
    }
    }
    
    .info-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 165, 0, 0.3);
    }
    
    .info-card p {
        margin: 12px 0;
        font-size: 1.05rem;
        font-weight: 500;
    }

    /* Price display */
    .price-display {
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.5) , transparent);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.3), inset 0 0 20px rgba(34, 197, 94, 0.1);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 30px rgba(34, 197, 94, 0.3), inset 0 0 20px rgba(34, 197, 94, 0.1); }
        50% { box-shadow: 0 0 40px rgba(34, 197, 94, 0.5), inset 0 0 30px rgba(34, 197, 94, 0.2); }
    }
    
    .price-display h2 {
        font-size: 2rem !important;
        margin: 0 0 12px 0 !important;
        color: #4ade80 !important;
        text-shadow: 0 4px 8px black
    }
    
    .price-display p {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #4ade80 !important;
        text-shadow: 0 4px 8px black;
    }

    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.5), transparent);
        margin: 24px 0;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #FFD700 !important;
    }

    /* Error and success messages */
    .stAlert {
        border-radius: 12px !important;
        border-left: 4px solid !important;
    }
    
    /* Searchbox styling */
    [data-testid="stSearchBox"] input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Map container */
    .folium-map {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------Layout --------------
st.markdown(
    "<h1 style='text-align: center;'>🚕 Taxi Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🗺️ Trip Details")

    start = st_searchbox(
        search_function=fetch_address_suggestions,
        placeholder="📍 From (address)",
        key="start_search",
        default_options=[
            "Centralstationen, Stockholm",
            "Göteborg Central, Göteborg",
            "Malmö Centralstation, Malmö",
            "Arlanda Airport, Stockholm",
            "Drottninggatan 1, Stockholm",
            "Avenyn, Göteborg",
        ],
    )

    destination = st_searchbox(
        search_function=fetch_address_suggestions,
        placeholder="🎯 Destination",
        key="dest_search",
        default_options=[
            "Göteborg Central, Göteborg",
            "Malmö Centralstation, Malmö",
            "Arlanda Airport, Stockholm",
            "Drottninggatan 1, Stockholm",
            "Avenyn, Göteborg",
        ],
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_date, col_time = st.columns(2)
    with col_date:
        date = st.date_input("📅 Date", datetime.date.today())
    with col_time:
        time = st.time_input("⏰ Time", value=datetime.time(8, 0))

    passengers = st.number_input("👥 Number of Passengers", 1, 4, 1)

    st.markdown('</div>', unsafe_allow_html=True)

    # Additional inputs
    st.subheader("⚙️ Additional Parameters")

    col_a, col_b = st.columns(2)
    with col_a:
        traffic = st.selectbox(
            "🚦 Traffic", ["Low", "Medium", "High"], index=1
        )

    with col_b:
        weather = st.selectbox("🌤️ Weather", ["Clear", "Rain", "Snow"], index=0)
    
    st.markdown('</div>', unsafe_allow_html=True)

    base_fare = 35.0
    per_km_rate = 12.0
    per_minute_rate = 3.0

    if st.button("🚀 Calculate Price", use_container_width=True):
        if not start or not destination:
            st.error("Please provide both start and destination addresses!")
        else:
            st.session_state["inputs"] = {
                "start": start,
                "destination": destination,
                "date": date,
                "time": time,
                "passengers": passengers,
                "traffic": traffic,
                "weather": weather,
                "base_fare": base_fare,
                "per_km_rate": per_km_rate,
                "per_minute_rate": per_minute_rate,
            }

with col2:
    if "inputs" in st.session_state:
        req = st.session_state["inputs"]
        with st.spinner("🔄 Calculating your trip..."):
            # Geocode addresses
            start_lat, start_lon = geocode_address(req["start"])
            end_lat, end_lon = geocode_address(req["destination"])

            if not start_lat or not end_lat:
                st.error("One or more addresses could not be found")
            else:
                # Get route info
                dist_km, dur_min, coords = get_route_info(
                    start_lat, start_lon, end_lat, end_lon
                )

                if not dist_km or not dur_min:
                    st.error("Could not calculate route")
                else:
                    # Format duration
                    hours = int(dur_min // 60)
                    minutes = int(dur_min % 60)
                    total_duration = (
                        f"{hours}h {minutes}min" if hours > 0 else f"{minutes} min"
                    )

                    # Get time of day and day of week
                    time_of_day = get_time_of_day(req["time"])
                    day_of_week = req["date"].strftime("%A")
                    day_of_week = (
                        "Weekend"
                        if day_of_week in ["Saturday", "Sunday"]
                        else "Weekday"
                    )

                    # Create payload for API
                    payload = {
                        "Trip_Distance_km": round(dist_km, 2),
                        "Trip_Duration_Minutes": round(dur_min, 1),
                        "Day_of_Week": day_of_week,
                        "Time_of_Day": time_of_day,
                        "Passenger_Count": req["passengers"],
                        "Base_Fare": req["base_fare"],
                        "Per_Km_Rate": req["per_km_rate"],
                        "Per_Minute_Rate": req["per_minute_rate"],
                        "Traffic_Conditions": req["traffic"],
                        "Weather": req["weather"],
                    }

                    # Call API
                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/predict",
                            json=payload,
                            timeout=10,
                        )

                        if response.ok:
                            result = response.json()
                            price = result.get("predicted_price", 0) * 10

                            # Price display
                            st.markdown(
                                f"""
                                <div class="price-display">
                                    <h2>💰 Estimated Price</h2>
                                    <p>{price:.2f} SEK</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # Info cards
                            col_info1, col_info2 = st.columns(2)

                            with col_info1:
                                st.markdown(
                                    f"""
                                    <div class="info-card">
                                        <p>📏 <b>Distance</b><br/>{dist_km:.1f} km</p>
                                        <p>⏱️ <b>Duration</b><br/>{total_duration}</p>
                                        <p>👥 <b>Passengers</b><br/>{req['passengers']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col_info2: 
                                st.markdown(
                                    f"""
                                    <div class="info-card">
                                        <p>🚦 <b>Traffic</b><br/>{req['traffic']}</p>
                                        <p>🌤️ <b>Weather</b><br/>{req['weather']}</p>
                                        <p>🕐 <b>Time</b><br/>{time_of_day} • {day_of_week}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        
                            # Display map with route
                            if coords:
                                st.subheader("🗺️ Your Route")
                                midpoint = [
                                    (start_lat + end_lat) / 2,
                                    (start_lon + end_lon) / 2,
                                ]
                                m = folium.Map(location=midpoint, zoom_start=5, tiles='CartoDB dark_matter')

                                # Add route line with gradient effect
                                folium.PolyLine(
                                    coords, 
                                    color="#FFD700", 
                                    weight=4, 
                                    opacity=0.8
                                ).add_to(m)

                                # Add markers
                                folium.Marker(
                                    [start_lat, start_lon],
                                    tooltip="🟢 Start",
                                    icon=folium.Icon(color="gray", icon="play", prefix='fa'),
                                ).add_to(m)

                                folium.Marker(
                                    [end_lat, end_lon],
                                    tooltip="🔴 Destination",
                                    icon=folium.Icon(color="orange", icon="flag-checkered", prefix='fa'),
                                ).add_to(m)

                                st_folium(m, width=700, height=250)
                        else:
                            st.error(
                                f"API Error: {response.status_code} - {response.text}"
                            )

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to API. Is the server running?")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ API request timed out.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    else:
        st.markdown(
            """
            <div class="glass-box" style="text-align: center; padding: 60px 20px;">
                <h2 style="color: #FFD700; font-size: 2.5rem; margin-bottom: 16px;">👈 Get Started</h2>
                <p style="font-size: 1.2rem; color: rgba(255,255,255,0.8);">
                    Enter your trip details and click<br/>
                    <b style="color: #FFD700;">Calculate Price</b> to see your estimate
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )