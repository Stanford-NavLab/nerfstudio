from math import radians, cos, sin, sqrt, atan2

import requests 
import urllib
import json


def get_elevation_usgs(lat, lon):
    url = 'https://epqs.nationalmap.gov/v1/json?'

    params = {
        'x': lon,
        'y': lat, 
        'units': 'Meters',
        'output': 'json'
    }
  
    full_url = url + urllib.parse.urlencode(params)
    print("Querying...", full_url)
    response = requests.get(full_url)
    data = json.loads(response.text)
    print("...Done")
    if 'value' not in data.keys():
        print(data['message'])
        raise ValueError
    return data['value']


def geodetic_to_enu(lat_q, lon_q, h_q, lat_c, lon_c, h_c):
    """
    Convert geodetic coordinates to East-North-Up (ENU) coordinates.
    
    :param lat_q: Query point latitude in degrees.
    :param lon_q: Query point longitude in degrees.
    :param h_q: Query point altitude in meters.
    :param lat_c: Center point latitude in degrees.
    :param lon_c: Center point longitude in degrees.
    :param h_c: Center point altitude in meters.
    :return: A tuple with ENU coordinates (east, north, up).
    """
    
    # Convert latitude and longitude from degrees to radians
    lat_c_rad = radians(lat_c)
    lon_c_rad = radians(lon_c)
    lat_q_rad = radians(lat_q)
    lon_q_rad = radians(lon_q)
    
    # Constants
    a = 6378137.0  # Earth's radius in meters
    e_sq = 6.69437999014e-3  # Earth's eccentricity squared
    
    # Calculate prime vertical radius of curvature
    N = a / sqrt(1 - e_sq * sin(lat_c_rad)**2)
    
    # Calculate cartesian coordinates for center point (reference point)
    x_c = (N + h_c) * cos(lat_c_rad) * cos(lon_c_rad)
    y_c = (N + h_c) * cos(lat_c_rad) * sin(lon_c_rad)
    z_c = ((1 - e_sq) * N + h_c) * sin(lat_c_rad)
    
    # Calculate cartesian coordinates for query point
    N = a / sqrt(1 - e_sq * sin(lat_q_rad)**2)
    x_q = (N + h_q) * cos(lat_q_rad) * cos(lon_q_rad)
    y_q = (N + h_q) * cos(lat_q_rad) * sin(lon_q_rad)
    z_q = ((1 - e_sq) * N + h_q) * sin(lat_q_rad)
    
    # Calculate the differences
    dx = x_q - x_c
    dy = y_q - y_c
    dz = z_q - z_c
    
    # Calculate ENU coordinates
    east = -sin(lon_c_rad) * dx + cos(lon_c_rad) * dy
    north = -sin(lat_c_rad) * cos(lon_c_rad) * dx - sin(lat_c_rad) * sin(lon_c_rad) * dy + cos(lat_c_rad) * dz
    up = cos(lat_c_rad) * cos(lon_c_rad) * dx + cos(lat_c_rad) * sin(lon_c_rad) * dy + sin(lat_c_rad) * dz
    
    return east, north, up

