"""Tools Claude can call while writing the report.

The Anthropic SDK's tool runner calls these functions when Claude asks for them.
Each one fetches live data the first time and caches the result in S3, so reruns
are fast and return the same data:
  - get_wind_data: the National Weather Service API finds the stations nearest the point,
    and the Iowa Environmental Mesonet archive has their past hourly observations
  - get_terrain: elevation and slope from the USGS elevation service
  - get_reference_fire_findings: Amazon Bedrock AgentCore Web Search
"""
import csv
import io
import json
import math
import os
import re
from datetime import date as Date, timedelta

import boto3
import requests
from anthropic import beta_tool

s3_client = boto3.client("s3")
web_search = boto3.client("bedrock-agentcore")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "")
CACHE_PREFIX = "reference-cache/"
MESONET = "https://mesonet.agron.iastate.edu"
NWS_HEADERS = {"User-Agent": "sam3-wildfire-workshop"}  # the NWS API asks callers to identify themselves


def _cached(key, fetch):
    """Return the cached S3 object at key, or call fetch() and cache its JSON result."""
    try:
        body = s3_client.get_object(Bucket=BUCKET_NAME, Key=CACHE_PREFIX + key)["Body"].read()
        return json.loads(body)
    except s3_client.exceptions.NoSuchKey:
        result = fetch()
        s3_client.put_object(Bucket=BUCKET_NAME, Key=CACHE_PREFIX + key, Body=json.dumps(result))
        return result


def _lat_lon(location):
    lat, lon = (float(v) for v in location.split(","))
    return round(lat, 4), round(lon, 4)


@beta_tool
def get_wind_data(location: str, date: str) -> str:
    """National Weather Service hourly wind speed, gusts and direction on a given date, from the
    weather station nearest a point in the US.

    Args:
        location: Latitude and longitude in decimal degrees, e.g. "34.045,-118.540".
        date: Date in YYYY-MM-DD format, e.g. "2025-01-07".
    """
    lat, lon = _lat_lon(location)
    day = Date.fromisoformat(date.strip())
    return json.dumps(_cached(f"wind-{lat}_{lon}-{day}.json", lambda: _wind(lat, lon, day)))


def _wind(lat, lon, day):
    # The NWS API gives the point's timezone and its observation stations, nearest first
    point = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=NWS_HEADERS, timeout=30).json()
    timezone = point["properties"]["timeZone"]
    stations = requests.get(point["properties"]["observationStations"], headers=NWS_HEADERS, timeout=30).json()

    end = day + timedelta(days=1)
    for station in stations["features"][:5]:  # use the nearest one with records for that day
        observations = requests.get(f"{MESONET}/cgi-bin/request/asos.py", timeout=30, params={
            "station": station["properties"]["stationIdentifier"], "data": ["sknt", "gust", "drct"],
            "report_type": "3",  # hourly reports
            "year1": day.year, "month1": day.month, "day1": day.day, "year2": end.year, "month2": end.month, "day2": end.day,
            "tz": timezone, "format": "onlycomma", "missing": "empty",
        }).text
        rows = list(csv.DictReader(io.StringIO(observations)))
        if rows and "sknt" in rows[0]:
            break
    else:
        return {"error": f"No hourly observations found near {lat},{lon} on {day}."}

    station_lon, station_lat = station["geometry"]["coordinates"]
    mph = lambda knots: round(float(knots) * 1.15078) if knots else None
    return {
        "station": f"{station['properties']['name']} ({station['properties']['stationIdentifier']})",
        "km_from_location": round(111.32 * math.hypot(station_lat - lat, (station_lon - lon) * math.cos(math.radians(lat))), 1),
        "note": f"Speeds in mph; direction in degrees the wind blew FROM (0 = north, 90 = east); times are {timezone}.",
        "hourly": [{"time": row["valid"][11:], "speed_mph": mph(row["sknt"]), "gust_mph": mph(row["gust"]),
                    "from_degrees": round(float(row["drct"])) if row["drct"] else None} for row in rows],
        "source": f"NWS/FAA observations via the Iowa Environmental Mesonet ({MESONET}); station from api.weather.gov",
    }


@beta_tool
def get_terrain(location: str) -> str:
    """Elevation and slope at a point, from the USGS 3D Elevation Program.

    Args:
        location: Latitude and longitude in decimal degrees, e.g. "34.045,-118.540".
    """
    lat, lon = _lat_lon(location)
    return json.dumps(_cached(f"terrain-{lat}_{lon}.json", lambda: _terrain(lat, lon)))


def _terrain(lat, lon):
    """Elevation at the point, and slope from the elevations ~30 m on each side of it."""
    def elevation(y, x):
        response = requests.get("https://epqs.nationalmap.gov/v1/json",
                                params={"x": x, "y": y, "units": "Meters", "wkid": 4326}, timeout=30)
        return float(response.json()["value"])

    step = 0.0003  # degrees, about 30 m
    meters = 2 * step * 111_320  # distance between the north and south points
    around = {"north": elevation(lat + step, lon), "south": elevation(lat - step, lon),
              "east": elevation(lat, lon + step), "west": elevation(lat, lon - step)}
    rise_north = (around["north"] - around["south"]) / meters
    rise_east = (around["east"] - around["west"]) / (meters * math.cos(math.radians(lat)))
    downhill = math.degrees(math.atan2(-rise_east, -rise_north)) % 360
    return {
        "elevation_m": round(elevation(lat, lon), 1),
        "elevation_30m_away_m": {side: round(value, 1) for side, value in around.items()},
        "slope_degrees": round(math.degrees(math.atan(math.hypot(rise_east, rise_north))), 1),
        "slope_faces": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(downhill / 45) % 8],
        "source": "USGS 3D Elevation Program (https://epqs.nationalmap.gov)",
    }


@beta_tool
def get_reference_fire_findings(fire: str, topic: str) -> str:
    """Published findings about homes in a past wildfire, from a web search. Returns the top results,
    each with a text extract from the page.

    Args:
        fire: The past wildfire, with its year, e.g. "2018 Camp Fire".
        topic: A topic such as "roofing", "home spacing", "vegetation" or "embers".
    """
    query = f"{fire} homes {topic} study findings"
    key = re.sub(r"[^a-z0-9]+", "-", query.lower())
    return json.dumps({
        "search_query": query,
        "results": _cached(f"findings-{key}.json", lambda: _search(query)),
        "note": "Extracts are copied from each page. Check each page's publication date, and cite the URL with any finding you use.",
    })


def _search(query):
    response = web_search.invoke_web_search(webSearchId="web-search.v1", searchQuery=query, maxResults=5)
    return [{"title": r.get("title"), "url": r.get("url"), "published": str(r.get("publishedAt", "unknown"))[:10],
             "extract": r["text"][:1200]} for r in response["results"]]


TOOLS = [get_wind_data, get_terrain, get_reference_fire_findings]
