import requests

def get_weather_data(latitude, longitude):
  """Fetches weather information for a given latitude and longitude."""

  # Get the grid forecast endpoint URL
  points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
  response = requests.get(points_url)
  response.raise_for_status()  # Raise an error for bad responses

  data = response.json()
  forecast_url = data["properties"]["forecast"]

  # Fetch the forecast data
  response = requests.get(forecast_url)
  response.raise_for_status()

  forecast = response.json()
  return forecast

# Example usage
latitude = 39.7456
longitude = -97.0892
weather_data = get_weather_data(latitude, longitude)

# Print the forecast
print(weather_data)
