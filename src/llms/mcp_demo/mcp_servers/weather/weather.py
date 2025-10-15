from typing import Any, Dict
import httpx
from mcp.server.fastmcp import FastMCP


# init mcp server
mcp = FastMCP("weather")

# Contents
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> Dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            return None
        

def format_alter(feature: Dict) -> str:
    """Format an alter feature into a readable string."""
    props = feature.get("properties", {})
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Serverity: {props.get("severity", "Unknown")}
Descriptoin: {props.get("description", "No Description available")}
Instruction: {props.get("instruction", "No Instruction provided")}
"""


# Implementing tool execution
@mcp.tool()
async def get_alters(state: str) -> str:
    """Get weather alters for a given state.
    
    Args:
        state (str): The state to get weather alters for.
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    
    if not data.get("features"):
        return "No active alerts for this state."
    
    alerts = [format_alter(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # 
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."
    
    # get the forecast url from the points response
    forecast_url = points_data.get("properties", {}).get("forecast")
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."
    
    # format
    periods = forecast_data.get("properties", {}).get("periods", [])
    forcasts = []
    for period in periods[:5]:
        forcast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forcasts.append(forcast)

    return "\n---\n".join(forcasts)


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
    