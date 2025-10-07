import streamlit as st
import datetime
from dotenv import load_dotenv
import requests
import pathlib
import base64
import polyline
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------- Background --------------
def set_bg(image_path: str):
    # Get directory of the file where this function lives
    base_dir = pathlib.Path(__file__).parent.resolve()

    # Build full path relative to this utils file
    full_path = (base_dir / image_path).resolve()

    if not full_path.exists():
        st.error(f"Background image not found: {full_path}")
        return

    img_bytes = full_path.read_bytes()
    taxi_image = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{taxi_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-color: transparent !important;
    }}
    .block-container {{
        max-width: 1200px;
        margin: auto;
        padding-top: 3rem;
    }}
   
    </style>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Taxi Prediction", layout="wide")


# ---------------- Funktioner ----------------
def geocode_address(address: str):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_API_KEY}
    resp = requests.get(url, params=params).json()
    if resp.get("results"):
        loc = resp["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None


def get_time_of_day(time: datetime.time) -> str:
    hour = time.hour
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"


def fetch_address_suggestions(searchterm):
    
    if not searchterm or len(searchterm) < 2:
        return ["Centralstationen, Stockholm",
            "Göteborg Central, Göteborg",
            "Malmö Centralstation, Malmö",
            "Arlanda Airport, Stockholm",
            "Drottninggatan 1, Stockholm",
            "Avenyn, Göteborg"]
    
    url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    params = {
        "input": searchterm,
        "types": "geocode",
        "components": "country:se",  #limit to sweden
        "language": "sv",
        "key": GOOGLE_API_KEY,
    }
    try:
        resp = requests.get(url, params=params).json()
        
        if resp.get('status') == 'OK':
            return [prediction['description'] for prediction in resp.get('predictions', [])]
        
        return []
        
    except Exception as e:
            print(f"Error fetching suggestions: {e}")
            return []


def get_route_info(start_lat, start_lon, end_lat, end_lon):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "mode": "driving",
        "key": GOOGLE_API_KEY,
    }
    resp = requests.get(url, params=params).json()
    if resp.get("routes"):
        leg = resp["routes"][0]["legs"][0]
        distance_km = leg["distance"]["value"] / 1000
        duration_min = leg["duration"]["value"] / 60

        # Hämta polyline för rutten
        points = resp["routes"][0]["overview_polyline"]["points"]
        coords = polyline.decode(points)  # lista med (lat, lon)

        return distance_km, duration_min, coords
    return None, None, []
