from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import json
import os
from pathlib import Path

app = Flask(__name__)

# Configuration
DATA_DIR = Path("data")
CORRECTIONS_DIR = Path("corrections")
CORRECTIONS_FILE = CORRECTIONS_DIR / "station_corrections.json"

# Create corrections directory if it doesn't exist
CORRECTIONS_DIR.mkdir(exist_ok=True)

# Initialize corrections file if it doesn't exist
if not CORRECTIONS_FILE.exists():
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump({}, f)

@app.route('/')
def index():
    # Get list of all available year_side combinations
    year_sides = []
    for item in os.listdir(DATA_DIR):
        item_path = DATA_DIR / item
        if item_path.is_dir() and (item_path / "stops.csv").exists():
            year, side = item.split('_') if '_' in item else (item, None)
            year_sides.append({"year": year, "side": side, "id": item})
    
    return render_template('index.html', year_sides=sorted(year_sides, key=lambda x: (x['year'], x['side'])))

@app.route('/data/<year_side>')
def get_year_side_data(year_side):
    """Get all data for a specific year_side"""
    stops_file = DATA_DIR / year_side / "stops.csv"
    lines_file = DATA_DIR / year_side / "lines.csv"
    line_stops_file = DATA_DIR / year_side / "line_stops.csv"
    
    # Load corrected stations
    corrections = {}
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, 'r') as f:
            corrections = json.load(f)
    
    # Read data
    try:
        stops_df = pd.read_csv(stops_file)
        lines_df = pd.read_csv(lines_file)
        line_stops_df = pd.read_csv(line_stops_file)
        
        # Apply corrections to stops dataframe (only for display, not modifying original)
        stops_display = stops_df.copy()
        for stop_id, correction in corrections.get(year_side, {}).items():
            if stop_id in stops_display['stop_id'].values:
                idx = stops_display.loc[stops_display['stop_id'] == stop_id].index[0]
                stops_display.at[idx, 'location'] = f"{correction['lat']},{correction['lng']}"
        
        # Convert to GeoJSON for mapping
        features = []
        for _, row in stops_display.iterrows():
            if pd.notna(row['location']):
                try:
                    lat, lng = map(float, str(row['location']).strip('"').split(','))
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lng, lat]  # GeoJSON uses [lng, lat] order
                        },
                        "properties": {
                            "stop_id": row['stop_id'],
                            "name": row['stop_name'],
                            "type": row['type'],
                            "line": row['line_name'],
                            "corrected": str(row['stop_id']) in corrections.get(year_side, {})
                        }
                    }
                    features.append(feature)
                except Exception as e:
                    print(f"Error with stop {row['stop_id']}: {e}")
        
        # Get line data
        lines = []
        for _, row in lines_df.iterrows():
            lines.append({
                "line_id": row['line_id'],
                "line_name": row['line_name'],
                "type": row['type']
            })
            
        return jsonify({
            "geojson": {"type": "FeatureCollection", "features": features},
            "lines": lines
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_correction', methods=['POST'])
def save_correction():
    """Save a station location correction"""
    data = request.json
    year_side = data['year_side']
    stop_id = str(data['stop_id'])  # Ensure stop_id is a string
    lat = data['lat']
    lng = data['lng']
    
    # Load existing corrections
    with open(CORRECTIONS_FILE, 'r') as f:
        corrections = json.load(f)
    
    # Add or update correction
    if year_side not in corrections:
        corrections[year_side] = {}
    
    corrections[year_side][stop_id] = {"lat": lat, "lng": lng}
    
    # Save corrections
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f, indent=2)
    
    return jsonify({"status": "success"})

@app.route('/export_corrections')
def export_corrections():
    """Export corrected data as CSVs"""
    # Load corrections
    with open(CORRECTIONS_FILE, 'r') as f:
        corrections = json.load(f)
    
    # Create export directory
    export_dir = Path("export")
    export_dir.mkdir(exist_ok=True)
    
    # For each year_side with corrections
    for year_side, stops_corrections in corrections.items():
        # Read original stops file
        stops_file = DATA_DIR / year_side / "stops.csv"
        if not stops_file.exists():
            continue
        
        stops_df = pd.read_csv(stops_file)
        
        # Apply corrections
        for stop_id, correction in stops_corrections.items():
            if int(stop_id) in stops_df['stop_id'].values:
                idx = stops_df.loc[stops_df['stop_id'] == int(stop_id)].index[0]
                stops_df.at[idx, 'location'] = f"{correction['lat']},{correction['lng']}"
        
        # Save corrected file
        year_side_dir = export_dir / year_side
        year_side_dir.mkdir(exist_ok=True)
        stops_df.to_csv(year_side_dir / "stops.csv", index=False)
        
        # Copy other files without modification
        for filename in ["lines.csv", "line_stops.csv"]:
            src_file = DATA_DIR / year_side / filename
            if src_file.exists():
                pd.read_csv(src_file).to_csv(year_side_dir / filename, index=False)
    
    return jsonify({"status": "success", "message": "Data exported to 'export' directory"})

@app.route('/get_corrections')
def get_corrections():
    """Get all corrections"""
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, 'r') as f:
            corrections = json.load(f)
        return jsonify(corrections)
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)