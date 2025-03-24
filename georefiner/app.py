import os
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import json
from utils.geo_utils import get_map_bounds, create_geojson_from_stations
from utils.db_utils import load_stations_by_year, save_station_updates
from config import Config
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def index():
    """Main page of the application"""
    # Get available years for which we have data
    available_years = []
    for folder in os.listdir(app.config['PROCESSED_DATA_DIR']):
        folder_path = os.path.join(app.config['PROCESSED_DATA_DIR'], folder)
        if os.path.isdir(folder_path):
            # Check if the folder starts with a year (YYYY_)
            parts = folder.split('_')
            if len(parts) >= 1:
                try:
                    year = int(parts[0])
                    if year not in available_years:
                        available_years.append(year)
                except ValueError:
                    continue
    
    available_years.sort()
    
    # Get available map layers
    map_layers = []
    for file in os.listdir(app.config['MAPS_DIR']):
        if file.endswith(('.tif', '.geotiff')):
            map_layers.append(file)
    
    return render_template('index.html', 
                          available_years=available_years,
                          map_layers=map_layers)

@app.route('/api/stations/<int:year>')
def get_stations(year):
    """API endpoint to get stations for a specific year"""
    filter_type = request.args.get('type', None)
    filter_east_west = request.args.get('east_west', None)
    filter_line = request.args.get('line', None)
    
    stations = load_stations_by_year(year, filter_type, filter_east_west, filter_line)
    geojson = create_geojson_from_stations(stations)
    
    return jsonify(geojson)
    
@app.route('/api/lines/<int:year>')
def get_lines(year):
    """API endpoint to get available lines for a specific year"""
    # Load all stations for the year
    stations = load_stations_by_year(year)
    
    # Extract unique line names
    line_names = set()
    for station in stations:
        if 'line_name' in station and station['line_name']:
            line_names.add(station['line_name'])
    
    # Sort the lines
    sorted_lines = sorted(list(line_names))
    
    return jsonify(sorted_lines)

@app.route('/api/types/<int:year>')
def get_types(year):
    """API endpoint to get available station types for a specific year"""
    # Load all stations for the year
    stations = load_stations_by_year(year)
    
    # Extract unique types
    types = set()
    for station in stations:
        if 'type' in station and station['type'] is not None:
            type_val = str(station['type']).strip()
            if type_val:  # Only add non-empty values
                types.add(type_val)
    
    # Sort the types
    sorted_types = sorted(list(types))
    print(f"Found {len(sorted_types)} types for year {year}: {sorted_types}")
    
    return jsonify(sorted_types)

@app.route('/api/filter-options/<int:year>')
def get_filter_options(year):
    """API endpoint to get all filter options for a specific year in a single call"""
    # Print request info for debugging
    print(f"Filter options request for year {year} from {request.remote_addr}")
    
    # Load all stations for the year
    stations = load_stations_by_year(year)
    
    # Extract unique types and line names
    types = set()
    line_names = set()
    
    for station in stations:
        # Types
        if 'type' in station and station['type'] is not None:
            type_val = str(station['type']).strip()
            if type_val:  # Only add non-empty values
                types.add(type_val)
        
        # Line names
        if 'line_name' in station and station['line_name'] is not None:
            line_val = str(station['line_name']).strip()
            if line_val:  # Only add non-empty values
                line_names.add(line_val)
    
    # Sort the lists
    sorted_types = sorted(list(types))
    sorted_lines = sorted(list(line_names))
    
    print(f"For year {year}:")
    print(f"Found {len(sorted_types)} types: {sorted_types}")
    print(f"Found {len(sorted_lines)} lines: {sorted_lines}")
    
    # Create the response
    response = jsonify({
        "year": year,
        "types": sorted_types,
        "lines": sorted_lines
    })
    
    # Add cache-control headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

@app.route('/api/map/<path:filename>')
def serve_map(filename):
    """Serve map files directly"""
    print(f"Serving map file: {filename}")
    response = send_from_directory(app.config['MAPS_DIR'], filename, as_attachment=False)
    # Print response headers for debugging
    print(f"Response headers: {response.headers}")
    return response

@app.route('/api/map_metadata/<path:filename>')
def get_map_metadata(filename):
    """Get metadata for a specific map file"""
    map_path = os.path.join(app.config['MAPS_DIR'], filename)
    if not os.path.exists(map_path):
        return jsonify({"error": "Map file not found"}), 404
    
    bounds = get_map_bounds(map_path)
    return jsonify({
        "bounds": bounds,
        "filename": filename
    })

@app.route('/api/update_station', methods=['POST'])
def update_station():
    """Update a station's geolocation"""
    data = request.json
    
    station_id = data.get('id')
    new_lat = data.get('lat')
    new_lng = data.get('lng')
    year = data.get('year')
    
    if not all([station_id, new_lat, new_lng, year]):
        return jsonify({"status": "error", "message": "Missing required data"}), 400
    
    try:
        success = save_station_updates(station_id, float(new_lat), float(new_lng), int(year))
        if success:
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to update station"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/export_geojson/<int:year>')
def export_geojson(year):
    """API endpoint to export stations as GeoJSON"""
    from utils.db_utils import export_stations_geojson
    
    output_path = export_stations_geojson(year)
    if not output_path:
        return jsonify({"error": "No data found for this year"}), 404
    
    # Create a response with the file
    return send_file(output_path, as_attachment=True, 
                     download_name=f'stations_{year}.geojson',
                     mimetype='application/geo+json')

@app.route('/debug-api/types/<int:year>')
def debug_types(year):
    """Debugging endpoint to see what types are available"""
    stations = load_stations_by_year(year)
    
    # Extract unique types and count occurrences
    type_counts = {}
    for station in stations:
        if 'type' in station and station['type']:
            type_counts[station['type']] = type_counts.get(station['type'], 0) + 1
    
    return jsonify({
        "year": year,
        "total_stations": len(stations),
        "type_counts": type_counts
    })

@app.route('/debug-api/file/<path:filepath>')
def debug_file(filepath):
    """Debugging endpoint to view a CSV file"""
    try:
        # Ensure the path is within the data directory
        if not filepath.startswith(app.config['DATA_DIR']):
            full_path = os.path.join(app.config['DATA_DIR'], filepath)
        else:
            full_path = filepath
            
        if not os.path.exists(full_path):
            return jsonify({"error": f"File {full_path} not found"}), 404
        
        if full_path.endswith('.csv'):
            # For CSV files, return parsed content
            try:
                df = pd.read_csv(full_path, encoding='utf-8-sig')
                return jsonify({
                    "filename": filepath,
                    "columns": df.columns.tolist(),
                    "rows": df.head(10).to_dict('records'),
                    "total_rows": len(df)
                })
            except Exception as e:
                return jsonify({"error": f"Error parsing CSV: {str(e)}"}), 500
        else:
            # For other file types, return file info
            return jsonify({
                "filename": filepath,
                "size": os.path.getsize(full_path),
                "modified": os.path.getmtime(full_path)
            })
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/debug-api/folder/<path:folderpath>')
def debug_folder(folderpath):
    """Debugging endpoint to list contents of a folder"""
    try:
        # Ensure the path is within the data directory
        if not folderpath.startswith(app.config['DATA_DIR']):
            full_path = os.path.join(app.config['DATA_DIR'], folderpath)
        else:
            full_path = folderpath
            
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return jsonify({"error": f"Folder {full_path} not found or is not a directory"}), 404
        
        files = []
        folders = []
        
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                folders.append({
                    "name": item,
                    "path": os.path.join(folderpath, item),
                    "modified": os.path.getmtime(item_path)
                })
            else:
                files.append({
                    "name": item,
                    "path": os.path.join(folderpath, item),
                    "size": os.path.getsize(item_path),
                    "modified": os.path.getmtime(item_path)
                })
        
        return jsonify({
            "path": folderpath,
            "folders": folders,
            "files": files
        })
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/api-test')
def api_test_page():
    """Serve a simple HTML page for testing the API directly"""
    with open('direct-api-test.html', 'r') as file:
        content = file.read()
    return content

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(app.config['PROCESSED_DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['MAPS_DIR'], exist_ok=True)
    
    # Run the application
    app.run(debug=True)