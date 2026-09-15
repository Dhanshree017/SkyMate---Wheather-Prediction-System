import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file if running locally
load_dotenv()


class WeatherAPIClient:
    def __init__(self, api_key: str = None):
        # Fetch API key from parameter or environment variable
        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
        self.base_url = "http://api.weatherapi.com/v1"

        if not self.api_key:
            raise ValueError(
                "API Key not found! Please set 'WEATHER_API_KEY' in your environment or .env file."
            )

    def get_live_weather(self, location: str) -> dict:
        """Fetches current live weather data for a given location (City, Zip code, or Lat/Lon)."""
        endpoint = f"{self.base_url}/current.json"
        params = {
            "key": self.api_key,
            "q": location,
            "aqi": "yes"  # Include Air Quality data matching your dataset schema
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to fetch weather data. HTTP {response.status_code}: {response.text}"
            )

    def extract_features(self, live_json: dict) -> dict:
        """Extracts and maps raw API JSON response into features compatible with model training."""
        current = live_json.get("current", {})
        location = live_json.get("location", {})
        air_quality = current.get("air_quality", {})

        extracted_data = {
            # Location features
            "country": location.get("country"),
            "location_name": location.get("name"),
            "region": location.get("region"),
            "latitude": location.get("lat"),
            "longitude": location.get("lon"),

            # Weather features
            "humidity": current.get("humidity"),
            "pressure_mb": current.get("pressure_mb"),
            "wind_mph": current.get("wind_mph"),
            "wind_kph": current.get("wind_kph"),
            "wind_degree": current.get("wind_degree"),
            "wind_dir": current.get("wind_dir"),
            "precip_mm": current.get("precip_mm"),
            "cloud": current.get("cloud"),
            "feelslike_c": current.get("feelslike_c"),
            "vis_km": current.get("vis_km"),
            "uv_index": current.get("uv"),
            "gust_kph": current.get("gust_kph"),

            # Air Quality features
            "air_quality_Carbon_Monoxide": air_quality.get("co"),
            "air_quality_Ozone": air_quality.get("o3"),
            "air_quality_Nitrogen_dioxide": air_quality.get("no2"),
            "air_quality_Sulphur_dioxide": air_quality.get("so2"),
            "air_quality_PM2.5": air_quality.get("pm2_5"),
            "air_quality_PM10": air_quality.get("pm10"),
            "air_quality_us-epa-index": air_quality.get("us-epa-index"),
            "air_quality_gb-defra-index": air_quality.get("gb-defra-index")
        }

        return extracted_data


if __name__ == "__main__":
    # Test client execution locally
    try:
        client = WeatherAPIClient()
        city_name = "Mumbai"
        print(f"Fetching real-time data for: {city_name}...")
        
        raw_weather = client.get_live_weather(city_name)
        features = client.extract_features(raw_weather)
        
        print("\nLive Weather Data Successfully Extracted!")
        print(f"Current Temp: {raw_weather['current']['temp_c']} °C")
        print(f"Extracted {len(features)} live features for model prediction.")
    except Exception as e:
        print(f"Error testing API Client: {e}")