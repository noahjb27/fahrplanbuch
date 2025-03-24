# Berlin Transport Geolocation Refiner

A Flask web application for refining and correcting the geolocation of Berlin's public transportation stations using historical map overlays.

## Features

- View stations on historical map overlays for specific years
- Filter stations by type, line, and east/west location
- Interactive dragging to refine station positions
- Save corrections back to your data files
- Export GeoJSON for use in GIS applications

## Setup

### Prerequisites

- Python 3.8 or higher
- GDAL/OGR libraries for geospatial processing

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd berlin-transport-geo-refiner
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install GDAL dependencies (for Ubuntu/WSL):
   ```
   sudo apt-get update
   sudo apt-get install python3-gdal gdal-bin libgdal-dev
   ```

5. Prepare your data:
   - Create the `data/maps` directory and add your TIF files
   - Create the `data/processed` directory with subdirectories for each year and side (e.g., `data/processed/1965_west`)
   - For each year/side, add station data in a file named `stops.csv`

### Data Structure

The application expects your data to be organized as follows:

```
data/
├── maps/
│   └── 1965_west_liniennetz_modified.tif
└── processed/
    ├── 1965_west/
    │   └── stops.csv
    └── 1965_east/
        └── stops.csv
```

### CSV Format

The stops.csv file should include columns for:
- `stop_name`: Name of the station
- `type`: Station type (u-bahn, s-bahn, bus, strassenbahn, etc.)
- `line_name`: Line designation (15, A1, etc.)
- `stop_id`: Unique identifier
- `location`: Coordinates in format "latitude,longitude"
- `east_west`: Location relative to Berlin Wall ("east" or "west")

Example:
```
stop_name,type,line_name,stop_id,location,identifier,neighbourhood,district,east_west,postal_code
"Marienfelde, Daimlerstrasse",tram,15,19650,"52.42393712,13.38022295",,Marienfelde,Tempelhof-Schöneberg,west,12277
```

### Running the Application

Start the Flask development server:

```
python app.py
```

The application will be available at http://127.0.0.1:5000/

## Usage Instructions

1. Select a year from the dropdown to load stations from that time period
2. Choose a historical map layer as a reference
3. Filter stations by type, line, or east/west location
4. Enable "Edit Mode" to make corrections
5. Click on a station to select it, then drag the marker to the correct position
6. Click "Save New Location" to update the station data

## Troubleshooting

If you encounter issues with GeoTIFF files not displaying correctly:

1. Check if your TIF files are properly georeferenced using the included tool:
   ```
   python utils/check_tiff.py data/maps/your_file.tif
   ```

2. If necessary, add georeferencing with GDAL:
   ```
   gdal_translate -a_srs EPSG:4326 data/maps/your_file.tif data/maps/your_file_georeferenced.tif
   ```

3. For testing the API directly:
   ```
   python test_api.py 1965
   ```
   