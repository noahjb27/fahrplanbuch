import os
import json
import math
import rasterio
from rasterio.warp import transform_bounds
from pyproj import Transformer
import pyproj

def get_map_bounds(map_path):
    """
    Get the bounds of a GeoTIFF map file in [west, south, east, north] format
    
    Args:
        map_path: Path to the map file
        
    Returns:
        List with [west, south, east, north] coordinates in WGS84
    """
    try:
        with rasterio.open(map_path) as src:
            # Print raster info for debugging
            print(f"Raster CRS: {src.crs}")
            print(f"Raster bounds: {src.bounds}")
            
            # Get bounds in the original CRS
            bounds = src.bounds
            
            # Transform to WGS84 (EPSG:4326) if the source is in a different CRS
            if src.crs and src.crs != 'EPSG:4326':
                try:
                    bounds = transform_bounds(src.crs, 'EPSG:4326', 
                                          bounds.left, bounds.bottom, 
                                          bounds.right, bounds.top)
                except Exception as e:
                    print(f"Error transforming bounds: {e}")
                    # Return a default bounding box if transformation fails
                    return [13.0884, 52.3382, 13.7613, 52.6755]
            
            # Return as [west, south, east, north] for Leaflet
            return [bounds[0], bounds[1], bounds[2], bounds[3]]
    except Exception as e:
        print(f"Error getting map bounds: {e}")
        # Return a default bounding box for Berlin if we can't read the file
        return [13.0884, 52.3382, 13.7613, 52.6755]

def create_geojson_from_stations(stations):
    """
    Create a GeoJSON FeatureCollection from a list of station dictionaries
    
    Args:
        stations: List of station dictionaries
        
    Returns:
        GeoJSON FeatureCollection dictionary
    """
    features = []
    
    for station in stations:
        # Skip stations without coordinates
        lat = station.get('latitude')
        lon = station.get('longitude')
        
        # Check for different possible column names
        if lat is None and 'lat' in station:
            lat = station.get('lat')
        if lon is None and 'lng' in station:
            lon = station.get('lng')
        if lon is None and 'lon' in station:
            lon = station.get('lon')
            
        # Check for 'location' field in your custom format "lat,lng"
        if (lat is None or lon is None) and 'location' in station:
            try:
                location = station['location']
                # Remove quotes if present
                if isinstance(location, str):
                    location = location.strip('"')
                    coords = location.split(',')
                    if len(coords) == 2:
                        lat = float(coords[0])
                        lon = float(coords[1])
            except:
                pass
            
        # Also check for coordinate as a string with format "lat,lon"
        if (lat is None or lon is None) and 'coordinate_location' in station:
            try:
                coords = station['coordinate_location'].split(',')
                if len(coords) == 2:
                    lat = float(coords[0])
                    lon = float(coords[1])
            except:
                pass
        
        if lat is None or lon is None:
            continue
            
        # Convert to float if they're strings
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            continue
            
        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]  # GeoJSON uses [lon, lat] order
            },
            "properties": {}
        }
        
        # Include all station properties, but sanitize values
        for key, value in station.items():
            # Skip the coordinates as we've already processed them
            if key in ['latitude', 'longitude', 'lat', 'lng', 'lon', 'location', 'coordinate_location']:
                continue
                
            # Handle NaN values which cause JSON serialization errors
            if isinstance(value, float) and math.isnan(value):
                feature["properties"][key] = None
            elif value == 'NaN' or value == 'nan':
                feature["properties"][key] = None
            else:
                feature["properties"][key] = value
        
        features.append(feature)
    
    # Create the FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson
    


def reproject_point(x, y, from_epsg, to_epsg=4326):
    """
    Reproject a point from one coordinate system to another
    
    Args:
        x: X coordinate (longitude or easting)
        y: Y coordinate (latitude or northing)
        from_epsg: Source EPSG code
        to_epsg: Target EPSG code (defaults to WGS84)
        
    Returns:
        Tuple with (x, y) in the target coordinate system
    """
    transformer = Transformer.from_crs(f"EPSG:{from_epsg}", f"EPSG:{to_epsg}", always_xy=True)
    return transformer.transform(x, y)

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points in kilometers
    
    Args:
        lat1, lon1: Coordinates of the first point
        lat2, lon2: Coordinates of the second point
        
    Returns:
        Distance in kilometers
    """
    # Use the geodesic distance calculation from pyproj
    geod = pyproj.Geod(ellps='WGS84')
    
    # Forward azimuth, back azimuth, distance
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    
    # Convert from meters to kilometers
    return distance / 1000