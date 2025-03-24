import os
import json
import csv
import pandas as pd
from config import Config

def load_stations_by_year(year, filter_type=None, filter_east_west=None, filter_line=None):
    """
    Load station data for a specific year from CSV files.
    Assumes folder structure of YEAR_SIDE (e.g., "1965_west").
    
    Args:
        year: The year to load stations for
        filter_type: Optional filter for station type
        filter_east_west: Optional filter for east/west location
        filter_line: Optional filter for line_name
        
    Returns:
        List of station dictionaries
    """
    # Get all folders for the specified year
    all_stations = []
    
    # Look for folders that start with the year
    year_prefix = f"{year}_"
    found_folders = []
    
    for folder in os.listdir(Config.PROCESSED_DATA_DIR):
        folder_path = os.path.join(Config.PROCESSED_DATA_DIR, folder)
        
        if os.path.isdir(folder_path) and folder.startswith(year_prefix):
            found_folders.append(folder)
            # This is a folder for our year (like "1965_west")
            stops_file = os.path.join(folder_path, "stops.csv")
            
            if os.path.exists(stops_file):
                # Load the stops data
                try:
                    # Use encoding='utf-8-sig' to handle BOM if present
                    df = pd.read_csv(stops_file, encoding='utf-8-sig')
                    
                    # Print sample of the data for debugging
                    print(f"Sample data from {folder}:")
                    print(df.head(2))
                    print(f"Columns: {df.columns.tolist()}")
                    
                    # Convert NaN values to None for better JSON serialization
                    df = df.replace({pd.NA: None, float('nan'): None})
                    
                    # If line_name is numeric, convert to string
                    if 'line_name' in df.columns:
                        df['line_name'] = df['line_name'].astype(str)
                    
                    # If east_west isn't in the data but in the folder name, add it
                    if 'east_west' not in df.columns and '_' in folder:
                        east_west = folder.split('_')[1]
                        df['east_west'] = east_west
                    
                    # Extract side from folder name if needed
                    current_side = folder.split('_')[1] if '_' in folder else None
                    
                    # Add data from this file
                    stations = df.to_dict('records')
                    all_stations.extend(stations)
                except Exception as e:
                    print(f"Error loading {stops_file}: {e}")
    
    print(f"Found {len(found_folders)} folders for year {year}: {found_folders}")
    print(f"Loaded {len(all_stations)} stations total")
    
    # Apply filters if specified
    if filter_type:
        all_stations = [s for s in all_stations if s.get('type') == filter_type]
    
    if filter_east_west:
        all_stations = [s for s in all_stations if s.get('east_west') == filter_east_west]
    
    if filter_line:
        all_stations = [s for s in all_stations if str(s.get('line_name')) == filter_line]
    
    return all_stations

def save_station_updates(station_id, new_lat, new_lng, year):
    """
    Save updated station geolocation.
    This implementation saves to the original data files based on your structure.
    
    Args:
        station_id: ID of the station to update
        new_lat: New latitude
        new_lng: New longitude
        year: Year the station belongs to
        
    Returns:
        Boolean indicating success
    """
    # Find which subfolder the station belongs to
    year_prefix = f"{year}_"
    found_file = None
    station_east_west = None
    
    # Look through all folders that match the year
    for folder in os.listdir(Config.PROCESSED_DATA_DIR):
        folder_path = os.path.join(Config.PROCESSED_DATA_DIR, folder)
        
        if os.path.isdir(folder_path) and folder.startswith(year_prefix):
            stops_file = os.path.join(folder_path, "stops.csv")
            
            if os.path.exists(stops_file):
                # Check if this file contains our station
                df = pd.read_csv(stops_file)
                
                if 'stop_id' in df.columns:
                    # Convert to string for comparison
                    df['stop_id'] = df['stop_id'].astype(str)
                    if str(station_id) in df['stop_id'].values:
                        found_file = stops_file
                        station_east_west = folder.split('_')[1] if '_' in folder else None
                        break
    
    if not found_file:
        return False
    
    # Update the CSV file
    df = pd.read_csv(found_file)
    df['stop_id'] = df['stop_id'].astype(str)
    
    # Find the row index for our station
    idx = df[df['stop_id'] == str(station_id)].index
    
    if len(idx) == 0:
        return False
    
    # Update the location field with the new coordinates
    new_location = f"{new_lat},{new_lng}"
    
    if 'location' in df.columns:
        df.at[idx[0], 'location'] = new_location
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        df.at[idx[0], 'latitude'] = new_lat
        df.at[idx[0], 'longitude'] = new_lng
    else:
        # If neither location nor lat/lng columns exist, add them
        df.at[idx[0], 'location'] = new_location
    
    # Save back to CSV
    df.to_csv(found_file, index=False)
    
    # Create a record of the change in a separate log file
    folder_path = os.path.dirname(found_file)
    updates_log = os.path.join(folder_path, 'geo_updates.csv')
    log_exists = os.path.exists(updates_log)
    
    with open(updates_log, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(['station_id', 'new_latitude', 'new_longitude', 'timestamp', 'east_west'])
        
        from datetime import datetime
        writer.writerow([station_id, new_lat, new_lng, datetime.now().isoformat(), station_east_west])
    
    return True

def export_stations_geojson(year, output_path=None):
    """
    Export stations for a year as GeoJSON for use in GIS applications
    
    Args:
        year: Year to export
        output_path: Path to save the file (defaults to exports directory)
        
    Returns:
        Path to the exported file
    """
    from utils.geo_utils import create_geojson_from_stations
    
    stations = load_stations_by_year(year)
    if not stations:
        return None
    
    geojson = create_geojson_from_stations(stations)
    
    if not output_path:
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        output_path = os.path.join(Config.EXPORT_DIR, f'stations_{year}.geojson')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=4)
    
    return output_path