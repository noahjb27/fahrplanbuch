from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import json
import os
from pathlib import Path
from flask_cors import CORS
import rasterio

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

# Define tiles directory
TILES_DIR = Path("tiles")
TILES_DIR.mkdir(exist_ok=True)

# Make sure the TIF files directory exists (renamed from 1965 to tif for clarity)
TIF_DIR = TILES_DIR / "tif"
TIF_DIR.mkdir(parents=True, exist_ok=True)

CORS(app)

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

@app.route('/tiles/<path:filepath>')
def serve_tile(filepath):
    """Serve a tile from the tiles directory"""
    return send_from_directory(str(TILES_DIR), filepath)

@app.route('/available_tile_sets')
def available_tile_sets():
    """List all available tile sets"""
    if not TILES_DIR.exists():
        return jsonify([])
    
    tile_sets = []
    
    # Look for directories containing tile data (zoom level directories)
    for item in os.listdir(TILES_DIR):
        item_path = TILES_DIR / item
        
        # Skip the original TIF files or directories
        if item.endswith(('.tif', '.tiff')) or item == "tif":  # Changed from 1965 to tif
            continue
            
        if item_path.is_dir():
            # Check if this looks like a valid tile directory (has zoom level subdirectories)
            zoom_dirs = [d for d in os.listdir(item_path) if d.isdigit()]
            
            if zoom_dirs:
                tile_sets.append({
                    "name": item,
                    "url": f"/tiles/{item}/{{z}}/{{x}}/{{y}}.png",
                    "zoom_levels": sorted([int(z) for z in zoom_dirs])
                })
    
    return jsonify(tile_sets)

@app.route('/list_tif_files')
def list_tif_files():
    """List all TIF files in the tiles/tif directory"""
    tif_dir = TILES_DIR / "tif"  # Changed from 1965 to tif
    
    if not tif_dir.exists():
        return jsonify({"error": "Directory not found"}), 404
    
    tif_files = []
    for item in os.listdir(tif_dir):
        if item.lower().endswith(('.tif', '.tiff')):
            tif_files.append(item)
    
    return jsonify({"tif_files": tif_files})

@app.route('/process_tif/<filename>', methods=['POST'])
def process_single_tif(filename):
    """Process a single TIF file"""
    from rasterio_tile_generator import generate_xyz_tiles_rasterio
    
    tif_path = TILES_DIR / "tif" / filename
    
    if not tif_path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    
    output_dir = TILES_DIR / tif_path.stem
    
    success = generate_xyz_tiles_rasterio(
        tiff_file=tif_path,
        output_dir=output_dir,
        min_zoom=10,
        max_zoom=16
    )
    
    if success:
        return jsonify({
            "status": "success",
            "message": f"Tiles generated for {filename}",
            "url": f"/tiles/{tif_path.stem}/{{z}}/{{x}}/{{y}}.png"
        })
    else:
        return jsonify({"error": "Failed to generate tiles"}), 500

@app.route('/debug_image_bounds')
def debug_image_bounds():
    """Debug endpoint to get the bounds of the image in various formats"""
    tif_path = TILES_DIR / "tif" / "1965_ost_liniennetz_geo.tif"
    
    if not tif_path.exists():
        return jsonify({"error": "TIF file not found"})
    
    try:
        with rasterio.open(tif_path) as src:
            # Get bounds in original CRS
            bounds = src.bounds
            
            # Convert to WGS84 (EPSG:4326)
            from pyproj import Transformer
            if src.crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                west_lng, south_lat = transformer.transform(bounds.left, bounds.bottom)
                east_lng, north_lat = transformer.transform(bounds.right, bounds.top)
            else:
                west_lng, south_lat, east_lng, north_lat = bounds
                
            # Calculate center
            center_lng = (west_lng + east_lng) / 2
            center_lat = (south_lat + north_lat) / 2
            
            # Calculate Berlin coordinates in the image's CRS
            berlin_lng, berlin_lat = 13.4050, 52.5200
            if src.crs.to_epsg() != 4326:
                transformer_berlin = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                berlin_x, berlin_y = transformer_berlin.transform(berlin_lng, berlin_lat)
            else:
                berlin_x, berlin_y = berlin_lng, berlin_lat
                
            # Calculate pixel coordinates for Berlin
            try:
                berlin_pixel_row, berlin_pixel_col = src.index(berlin_x, berlin_y)
                berlin_in_image = 0 <= berlin_pixel_row < src.height and 0 <= berlin_pixel_col < src.width
            except:
                berlin_pixel_row = berlin_pixel_col = None
                berlin_in_image = False
            
            return jsonify({
                "original_crs": str(src.crs),
                "original_bounds": {
                    "west": bounds.left,
                    "south": bounds.bottom,
                    "east": bounds.right,
                    "north": bounds.top
                },
                "wgs84_bounds": {
                    "west": west_lng,
                    "south": south_lat,
                    "east": east_lng,
                    "north": north_lat
                },
                "center": {
                    "lng": center_lng,
                    "lat": center_lat
                },
                "berlin": {
                    "lng": berlin_lng,
                    "lat": berlin_lat,
                    "x": berlin_x,
                    "y": berlin_y,
                    "pixel_row": berlin_pixel_row,
                    "pixel_col": berlin_pixel_col,
                    "in_image": berlin_in_image
                },
                "image_dimensions": {
                    "width": src.width,
                    "height": src.height
                }
            })
    except Exception as e:
        return jsonify({"error": str(e)})
    
    # Check what the min and max x/y values are for this zoom level
    z_dir = tile_dir / str(berlin_center["z"])
    if z_dir.exists():
        x_dirs = [d for d in os.listdir(z_dir) if os.path.isdir(z_dir / d)]
        x_min = min([int(x) for x in x_dirs]) if x_dirs else None
        x_max = max([int(x) for x in x_dirs]) if x_dirs else None
        
        y_values = []
        for x_dir in x_dirs:
            x_path = z_dir / x_dir
            y_files = [f[:-4] for f in os.listdir(x_path) if f.endswith('.png')]
            y_values.extend([int(y) for y in y_files])
        
        y_min = min(y_values) if y_values else None
        y_max = max(y_values) if y_values else None
    else:
        x_min = x_max = y_min = y_max = None
    
    return jsonify({
        "central_tiles": central_tiles,
        "zoom_level_exists": os.path.exists(z_dir),
        "x_range": {"min": x_min, "max": x_max},
        "y_range": {"min": y_min, "max": y_max},
    })

@app.route('/process_all_tifs', methods=['POST'])
def process_all_tifs():
    """Process all TIF files in the tif directory"""
    from rasterio_tile_generator import process_tif_directory
    
    # Make sure the directory exists
    tif_dir = TILES_DIR / "tif"
    if not tif_dir.exists() or not any(f.endswith(('.tif', '.tiff')) for f in os.listdir(tif_dir)):
        return jsonify({
            "status": "error",
            "message": "No TIF files found in the tiles/tif directory. Please add TIF files first."
        }), 400
    
    results = process_tif_directory(
        base_dir=TILES_DIR / "tif",
        output_base_dir=TILES_DIR,
        min_zoom=10,
        max_zoom=16
    )
    
    success_count = sum(1 for result in results.values() if result["success"])
    
    return jsonify({
        "status": "success",
        "message": f"Processed {len(results)} files. {success_count} succeeded.",
        "details": results
    })

if __name__ == '__main__':
    app.run(debug=True)