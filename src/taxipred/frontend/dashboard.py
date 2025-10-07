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

# ----------------Layout --------------
st.title("🚕 Taxi Price Predictor")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Trip Details")

    start = st_searchbox(
        search_function=fetch_address_suggestions,
        placeholder="From (address)",
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
        placeholder="Destination",
        key="dest_search",
        default_options=[
            "Göteborg Central, Göteborg",
            "Malmö Centralstation, Malmö",
            "Arlanda Airport, Stockholm",
            "Drottninggatan 1, Stockholm",
            "Avenyn, Göteborg",
        ],
    )

    date = st.date_input("Date", datetime.date.today())
    time = st.time_input("Time", value=datetime.time(8, 0))
    passengers = st.number_input("Number of Passengers", 1, 4, 1)

    # Additional inputs for model
    st.subheader("Additional Parameters")

    col_a, col_b = st.columns(2)
    with col_a:
        traffic = st.selectbox(
            "Traffic", ["Low", "Medium", "High"], index=1
        )

    with col_b:
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"], index=0)
        base_fare = 35.0
        per_km_rate = 12.0
        per_minute_rate = 3.0

    if st.button("🚀 Calculate", use_container_width=True):
        if not start or not destination:
            st.error("Provide start- & destination address!")
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
        with st.spinner("Fetching route and predicting price!"):
            # Geocode addresses
            start_lat, start_lon = geocode_address(req["start"])
            end_lat, end_lon = geocode_address(req["destination"])

            if not start_lat or not end_lat:
                st.error("One or more adresses was not able to be found")
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

                            # results
                            st.success(f"### 💰 Estimated Price: {price:.2f} SEK")

                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("📏 Distance", f"{dist_km:.1f} km")
                                st.metric("⏱️ Time", total_duration)
                            with col_info2:
                                st.metric("👥 Passangers", req["passengers"])
                                st.metric("🚦 Trafic", req["traffic"])

                            st.info(
                                f"🌤️ Weather: {req['weather']} | 🕐 {time_of_day} | 📅 {day_of_week}"
                            )

                            # Display map with route
                            if coords:
                                st.subheader("🗺️ Route")
                                midpoint = [
                                    (start_lat + end_lat) / 2,
                                    (start_lon + end_lon) / 2,
                                ]
                                m = folium.Map(location=midpoint, zoom_start=5)

                                # Add route line
                                folium.PolyLine(
                                    coords, color="red", weight=3, opacity=0.5
                                ).add_to(m)

                                # Add markers
                                folium.Marker(
                                    [start_lat, start_lon],
                                    tooltip="🟢 Start",
                                    icon=folium.Icon(color="green", icon=0),
                                ).add_to(m)

                                folium.Marker(
                                    [end_lat, end_lon],
                                    tooltip="🔴 Destination",
                                    icon=folium.Icon(color="red", icon=0),
                                ).add_to(m)

                                st_folium(m, width=600, height=250)
                        else:
                            st.error(
                                f"API Error: {response.status_code} - {response.text}"
                            )

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to API. Is server running?")
                    except requests.exceptions.Timeout:
                        st.error("API timed out.")
                    except Exception as e:
                        st.error(f"Some fuckery happend: {str(e)}")
