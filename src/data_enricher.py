# Berlin Transport Data Processing - 1965 West Berlin
# ===================================================
# This notebook processes the 1965 West Berlin data through the complete workflow:
# 1. Initial CSV processing
# 2. Geolocation matching
# 3. Administrative enrichment
# 4. Validation and quality checks

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import logging
import ast
from shapely.geometry import Point
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
YEAR = 1965
SIDE = "west"
DATA_DIR = Path('../data')
RAW_DIR = DATA_DIR / 'raw' / SIDE
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'
GEO_DATA_DIR = DATA_DIR / 'data-external'

# Create required directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR / 'stops_base', exist_ok=True)
os.makedirs(INTERIM_DIR / 'stops_matched', exist_ok=True)
os.makedirs(INTERIM_DIR / 'stops_geolocated', exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# U-Bahn line profile mappings
KLEINPROFIL = {'1', '2', '3', '4', 'A', 'A I', 'A II', 'A III', 'A1', 'A2', 'B', 'B I', 'B II', 'B III', 'B1', 'B2'}
GROSSPROFIL = {'5', '6', '7', '8', '9', 'C', 'C I', 'C II', 'D', 'E', 'G'}

# =====================
# STAGE 1: DATA LOADING
# =====================
logger.info("STAGE 1: Loading raw data")

def load_raw_data(year, side):
    """Load raw data from CSV file"""
    try:
        raw_file = f"../data/raw/{year}_{side}.csv"
        if not raw_file.exists():
            logger.error(f"Raw data file not found: {raw_file}")
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")
            
        raw_df = pd.read_csv(raw_file)
        logger.info(f"Loaded raw data: {len(raw_df)} lines")
        
        # Load existing stations for matching
        try:
            existing_stations = pd.read_csv(PROCESSED_DIR / 'existing_stations.csv')
            logger.info(f"Loaded existing stations: {len(existing_stations)} stations")
        except FileNotFoundError:
            logger.warning("No existing stations file found. Creating new reference file.")
            existing_stations = pd.DataFrame(columns=['stop_id', 'stop_name', 'type', 'location', 'in_lines', 'identifier'])
            
        return raw_df, existing_stations
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Load the data
raw_df, existing_stations_df = load_raw_data(YEAR, SIDE)

# ==========================
# STAGE 2: DATA PREPARATION
# ==========================
logger.info("STAGE 2: Preparing and cleaning data")

def clean_data(df):
    """Clean and standardize raw data"""
    # Clean string columns
    string_cols = ['line_name', 'type', 'stops', 'east-west']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace('\u00a0', ' ')  # Remove non-breaking spaces
            
    # Clean numeric columns
    df['year'] = YEAR
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce').fillna(0).astype(int)
    if 'Length (time)' in df.columns:
        df['Length (time)'] = pd.to_numeric(df['Length (time)'], errors='coerce').fillna(0).astype(int)
    
    # Clean stops data - standardize separators
    df['stops'] = df['stops'].apply(standardize_stops)
    
    return df

def standardize_stops(stops_text):
    """Standardize stop separators and clean formatting"""
    # Replace various dash types with standard dash
    stops_text = re.sub(r'[\u2013\u2014\u2212]', '-', stops_text)
    # Ensure consistent spacing around dashes
    stops_text = re.sub(r'\s*-\s*', ' - ', stops_text)
    # Remove double spaces
    stops_text = re.sub(r'\s+', ' ', stops_text)
    return stops_text.strip()
    
def extract_first_last_stops(stops_text):
    """Extract first and last stops from the stops text"""
    stations = stops_text.split(' - ')
    return f"{stations[0]}<> {stations[-1]}"

# Clean the data
clean_df = clean_data(raw_df)

# =============================================
# STAGE 3: CREATE LINE TABLE
# =============================================
logger.info("STAGE 3: Creating normalized line")

def create_line_table(df):
    """Create standardized line table"""
    line_df = pd.DataFrame({
        'line_id': [f"{YEAR}{i:04d}" for i in range(1, len(df) + 1)],
        'year': YEAR,
        'line_name': df['line_name'],
        'type': df['type'],
        'start_stop': df['stops'].apply(extract_first_last_stops),
        'Length (time)': df['Length (time)'] if 'Length (time)' in df.columns else 0,
        'Length (km)': df['Length (km)'] if 'Length (km)' in df.columns else None,
        'east_west': df['east-west'],
        'Frequency': df['Frequency']
    })
    
    # Add profile for U-Bahn lines
    line_df['profile'] = None
    for idx, row in line_df.iterrows():
        if row['type'] == 'u-bahn':
            if row['line_name'] in KLEINPROFIL:
                line_df.at[idx, 'profile'] = 'Kleinprofil'
            elif row['line_name'] in GROSSPROFIL:
                line_df.at[idx, 'profile'] = 'Großprofil'
    
    # Add capacity based on transport type and profile
    line_df['capacity'] = None
    for idx, row in line_df.iterrows():
        if row['type'] == 'u-bahn' and row['profile'] == 'Kleinprofil':
            line_df.at[idx, 'capacity'] = 750
        elif row['type'] == 'u-bahn' and row['profile'] == 'Großprofil':
            line_df.at[idx, 'capacity'] = 1000
        elif row['type'] == 's-bahn':
            line_df.at[idx, 'capacity'] = 1100
        elif row['type'] == 'strassenbahn':
            line_df.at[idx, 'capacity'] = 195
        elif row['type'] in ['bus', 'bus (Umlandlinie)']:
            line_df.at[idx, 'capacity'] = 100
        elif row['type'] == 'fähre' or row['type'] == 'FÃ¤hre':
            line_df.at[idx, 'capacity'] = 300
    
    return line_df


# Create the three tables
line_df = create_line_table(clean_df)

# Display summaries
print(f"Created {len(line_df)} lines")

# Save the base tables
INTERIM_DIR.mkdir(exist_ok=True)
(INTERIM_DIR / 'stops_base').mkdir(exist_ok=True)

line_df.to_csv(INTERIM_DIR / 'stops_base' / f'lines_{YEAR}_{SIDE}.csv', index=False)

logger.info("Base tables saved to interim/stops_base directory")

# =============================================
# STAGE 5: ADMINISTRATIVE ENRICHMENT
# =============================================
logger.info("STAGE 5: Adding administrative data")

# Load stations
matched_path = f"../data/interim/stops_verified/stops_{YEAR}_{SIDE}.csv"
final_stops = pd.read_csv(matched_path)

def load_district_data():
    """Load district boundary data"""
    try:
        districts_path = GEO_DATA_DIR / "lor_ortsteile.geojson"
        districts_gdf = gpd.read_file(districts_path)
        
        # Load West Berlin districts
        west_berlin_path = GEO_DATA_DIR / "West-Berlin-Ortsteile.json"
        with open(west_berlin_path, "r") as f:
            west_berlin_districts = json.load(f)["West_Berlin"]
            
        return districts_gdf, west_berlin_districts
    except Exception as e:
        logger.error(f"Error loading district data: {str(e)}")
        return None, None

def add_administrative_data(stops_df, districts_gdf, west_berlin_districts):
    """Add administrative boundary information to stops"""
    # Make a copy to avoid modifying the original
    result_df = stops_df.copy()
    
    # Filter to only stops with location data
    stops_with_location = result_df[result_df['location'].notna()]
    
    if stops_with_location.empty:
        logger.warning("No stops with location data found, skipping administrative enrichment")
        return result_df
    
    try:
        # Convert to GeoDataFrame
        geometry = stops_with_location['location'].apply(
            lambda x: Point(*reversed([float(c.strip()) for c in x.split(',')]))
        )
        stops_gdf = gpd.GeoDataFrame(stops_with_location, geometry=geometry, crs="EPSG:4326")
        
        # Ensure CRS matches
        if stops_gdf.crs != districts_gdf.crs:
            districts_gdf = districts_gdf.to_crs(stops_gdf.crs)
        
        # Perform spatial join
        joined = gpd.sjoin(
            stops_gdf, 
            districts_gdf[['geometry', 'BEZIRK', 'OTEIL']], 
            how="left", 
            predicate='within'
        )
        
        # Add east/west classification
        joined['east_west_admin'] = joined['OTEIL'].apply(
            lambda x: 'west' if pd.notna(x) and x in west_berlin_districts else 'east'
        )
        
        # Drop geometry column and index_right from join
        if 'geometry' in joined.columns:
            joined = joined.drop(columns=['geometry'])
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        # Update original DataFrame
        for idx, row in joined.iterrows():
            original_idx = stops_df[stops_df['stop_id'] == row['stop_id']].index[0]
            result_df.at[original_idx, 'district'] = row.get('BEZIRK')
            result_df.at[original_idx, 'neighborhood'] = row.get('OTEIL')
            result_df.at[original_idx, 'east_west_admin'] = row.get('east_west_admin')
        
        # Validate east/west classification
        for idx, row in result_df.iterrows():
            if pd.notna(row.get('east_west_admin')) and pd.notna(row.get('east_west')):
                if row['east_west_admin'] != row['east_west']:
                    logger.warning(f"Mismatched east/west: {row['stop_name']} - Source says {row['east_west']} but location is in {row['east_west_admin']}")
        
        return result_df
    except Exception as e:
        logger.error(f"Error adding administrative data: {str(e)}")
        return result_df

# Load district data
districts_gdf, west_berlin_districts = load_district_data()

# Add administrative data
if districts_gdf is not None and west_berlin_districts is not None:
    enriched_stops_df = add_administrative_data(final_stops, districts_gdf, west_berlin_districts)
    
    # Save the enriched stops
    (INTERIM_DIR / 'stops_geolocated').mkdir(exist_ok=True)
    enriched_stops_df.to_csv(INTERIM_DIR / 'stops_geolocated' / f'stops_{YEAR}_{SIDE}_enriched.csv', index=False)
    
    logger.info(f"Enriched stops saved to interim/stops_geolocated directory")
else:
    logger.warning("Could not load district data, skipping administrative enrichment")
    enriched_stops_df = final_stops

# =============================================
# STAGE 6: VALIDATION AND QUALITY CHECKS
# =============================================
logger.info("STAGE 6: Validation and quality checks")

def validate_stops(stops_df):
    """Perform validation and quality checks on stops data"""
    # Check for missing locations
    missing_locations = stops_df['location'].isna().sum()
    print(f"Stops missing location: {missing_locations} ({missing_locations/len(stops_df)*100:.1f}%)")
    
    # Check for missing district/neighborhood
    missing_district = stops_df['district'].isna().sum() if 'district' in stops_df.columns else 'N/A'
    print(f"Stops missing district: {missing_district}")
    
    # Check for inconsistent east/west classification
    if 'east_west' in stops_df.columns and 'east_west_admin' in stops_df.columns:
        mismatched = stops_df[
            (stops_df['east_west'] != stops_df['east_west_admin']) & 
            stops_df['east_west_admin'].notna()
        ]
        if not mismatched.empty:
            print(f"Stops with mismatched east/west classification: {len(mismatched)}")
    
    return True

def validate_line_stops(line_stops_df, line_df):
    """Validate line-stops relationships"""
    # Check for missing line IDs
    line_ids_in_line_df = set(line_df['line_id'])
    line_ids_in_line_stops = set(line_stops_df['line_id'])
    
    missing_line_ids = line_ids_in_line_stops - line_ids_in_line_df
    if missing_line_ids:
        print(f"Line IDs in line_stops but not in lines: {missing_line_ids}")
    
    # Check for stop order gaps
    line_continuity_issues = []
    
    for line_id in line_ids_in_line_df:
        line_stops = line_stops_df[line_stops_df['line_id'] == line_id].sort_values('stop_order')
        
        if not line_stops.empty:
            stop_orders = line_stops['stop_order'].values
            expected_orders = np.arange(stop_orders.min(), stop_orders.max() + 1)
            
            missing = set(expected_orders) - set(stop_orders)
            if missing:
                line_continuity_issues.append({
                    'line_id': line_id,
                    'issue': 'Gap in stop sequence',
                    'missing_orders': sorted(missing)
                })
    
    if line_continuity_issues:
        print(f"Line continuity issues: {len(line_continuity_issues)}")
        for issue in line_continuity_issues[:3]:  # Show first 3 issues
            print(f"  {issue['line_id']}: {issue['issue']} - Missing orders: {issue['missing_orders']}")
    
    # Check for terminal stations matching
    terminal_issues = []
    
    for _, line in line_df.iterrows():
        line_id = line['line_id']
        line_stops = line_stops_df[line_stops_df['line_id'] == line_id].sort_values('stop_order')
        
        if not line_stops.empty:
            start_stop, end_stop = line['start_stop'].split('<>')
            start_stop = start_stop.strip()
            end_stop = end_stop.strip()
            
            if line_stops.iloc[0]['stop_name'] != start_stop:
                terminal_issues.append({
                    'line_id': line_id,
                    'issue': 'First stop mismatch',
                    'expected': start_stop,
                    'found': line_stops.iloc[0]['stop_name']
                })
            
            if line_stops.iloc[-1]['stop_name'] != end_stop:
                terminal_issues.append({
                    'line_id': line_id,
                    'issue': 'Last stop mismatch',
                    'expected': end_stop,
                    'found': line_stops.iloc[-1]['stop_name']
                })
    
    if terminal_issues:
        print(f"Terminal station issues: {len(terminal_issues)}")
        for issue in terminal_issues[:3]:  # Show first 3 issues
            print(f"  {issue['line_id']}: {issue['issue']} - Expected: {issue['expected']}, Found: {issue['found']}")
    
    return True

# Run validation
print("\nValidating stops data:")
validate_stops(enriched_stops_df)

print("\nValidating line-stops relationships:")
validate_line_stops(line_stops_df, line_df)

# =============================================
# STAGE 7: FINALIZE DATA
# =============================================
logger.info("STAGE 7: Finalizing data")

def finalize_data(line_df, stops_df, line_stops_df):
    """Prepare final datasets for output"""
    # Make copies to avoid modifying originals
    final_line_df = line_df.copy()
    final_stops_df = stops_df.copy()
    final_line_stops_df = line_stops_df.copy()
    
    # Ensure consistent column names and formats
    
    # For lines table
    if 'profile' not in final_line_df.columns:
        final_line_df['profile'] = None
    
    # For stops table
    # Ensure in_lines is stored as string
    final_stops_df['in_lines'] = final_stops_df['in_lines'].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    
    # Sort tables
    final_line_df = final_line_df.sort_values('line_id')
    final_stops_df = final_stops_df.sort_values('stop_id')
    final_line_stops_df = final_line_stops_df.sort_values(['line_id', 'stop_order'])
    
    return final_line_df, final_stops_df, final_line_stops_df

# Finalize data
final_line_df, final_stops_df, final_line_stops_df = finalize_data(line_df, enriched_stops_df, line_stops_df)

# Save final data
PROCESSED_DIR.mkdir(exist_ok=True)
final_line_df.to_csv(PROCESSED_DIR / f'lines_{YEAR}_{SIDE}.csv', index=False)
final_stops_df.to_csv(PROCESSED_DIR / f'stops_{YEAR}_{SIDE}.csv', index=False)
final_line_stops_df.to_csv(PROCESSED_DIR / f'line_stops_{YEAR}_{SIDE}.csv', index=False)

logger.info(f"Final data saved to processed directory")

# =============================================
# STAGE 8: UPDATE REFERENCE DATA
# =============================================
logger.info("STAGE 8: Updating reference data")

def update_reference_data(new_stops_df, existing_stations_df):
    """Update the reference stations dataset with new geolocated stops"""
    # Make a copy of existing stations
    updated_stations = existing_stations_df.copy() if existing_stations_df is not None else pd.DataFrame()
    
    # Filter to stops with location
    new_stops_with_location = new_stops_df[new_stops_df['location'].notna()]
    
    # Create a set of existing stop identifiers (stop_name + type)
    if not updated_stations.empty:
        existing_identifiers = set(
            updated_stations.apply(lambda x: f"{x['stop_name']}|{x['type']}", axis=1)
        )
    else:
        existing_identifiers = set()
    
    # Add new stops that aren't in the existing stations
    new_stops = []
    updated_count = 0
    
    for _, stop in new_stops_with_location.iterrows():
        stop_identifier = f"{stop['stop_name']}|{stop['type']}"
        
        if stop_identifier not in existing_identifiers:
            # This is a new stop, add it
            new_stops.append(stop)
        else:
            # This is an existing stop, update location if it was missing
            matching_idx = updated_stations[
                (updated_stations['stop_name'] == stop['stop_name']) &
                (updated_stations['type'] == stop['type'])
            ].index
            
            if not matching_idx.empty:
                idx = matching_idx[0]
                if pd.isna(updated_stations.at[idx, 'location']):
                    updated_stations.at[idx, 'location'] = stop['location']
                    updated_count += 1
    
    # Convert new stops to DataFrame
    if new_stops:
        new_stops_df = pd.DataFrame(new_stops)
        
        # Ensure consistent columns with existing_stations
        if not updated_stations.empty:
            for col in updated_stations.columns:
                if col not in new_stops_df.columns:
                    new_stops_df[col] = None
            
            # Select only the columns in existing_stations
            new_stops_df = new_stops_df[updated_stations.columns]
        
        # Combine with existing stations
        updated_stations = pd.concat([updated_stations, new_stops_df], ignore_index=True)
    
    logger.info(f"Added {len(new_stops)} new stops to reference data")
    logger.info(f"Updated locations for {updated_count} existing stops")
    
    return updated_stations

# Update reference data
updated_reference = update_reference_data(final_stops_df, existing_stations_df)

# Save updated reference data
updated_reference.to_csv(PROCESSED_DIR / 'existing_stations.csv', index=False)

logger.info(f"Updated reference data saved to processed directory")

# =============================================
# STAGE 9: VISUALIZE RESULTS
# =============================================
logger.info("STAGE 9: Visualizing results")

def visualize_network(stops_df, line_stops_df, line_df, output_path=None):
    """Create a visualization of the network"""
    try:
        # Filter to stops with location
        stops_with_location = stops_df[stops_df['location'].notna()].copy()
        
        if stops_with_location.empty:
            logger.warning("No stops with location data found, skipping visualization")
            return False
        
        # Extract coordinates
        stops_with_location[['lat', 'lon']] = stops_with_location['location'].str.split(',', expand=True).astype(float)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot by transport type with different colors
        transport_colors = {
            'u-bahn': 'blue',
            's-bahn': 'green',
            'strassenbahn': 'red',
            'bus': 'orange',
            'fähre': 'purple'
        }
        
        for transport_type, color in transport_colors.items():
            type_stops = stops_with_location[stops_with_location['type'] == transport_type]
            if not type_stops.empty:
                plt.scatter(type_stops['lon'], type_stops['lat'], 
                           c=color, label=transport_type, alpha=0.7)
        
        # Draw lines between connected stops
        for line_id in line_df['line_id'].unique():
            # Get line info
            line_info = line_df[line_df['line_id'] == line_id].iloc[0]
            line_type = line_info['type']
            
            # Get stops for this line
            line_stops = line_stops_df[line_stops_df['line_id'] == line_id].sort_values('stop_order')
            
            # Get stop IDs
            stop_ids = line_stops['stop_id'].tolist()
            
            # Get coordinates for these stops
            line_points = stops_with_location[stops_with_location['stop_id'].isin(stop_ids)]
            
            # Skip if not enough points or missing coordinates
            if len(line_points) < 2:
                continue
                
            # Order points by stop_order
            ordered_points = []
            for stop_id in stop_ids:
                stop_point = line_points[line_points['stop_id'] == stop_id]
                if not stop_point.empty:
                    ordered_points.append(stop_point.iloc[0])
            
            if len(ordered_points) < 2:
                continue
                
            # Extract coordinates
            lons = [p['lon'] for p in ordered_points]
            lats = [p['lat'] for p in ordered_points]
            
            # Draw line with appropriate color and low alpha
            plt.plot(lons, lats, c=transport_colors.get(line_type, 'gray'), 
                    alpha=0.3, linewidth=1)
        
        # Add labels and legend
        plt.title(f'Berlin {SIDE.capitalize()} Transport Network ({YEAR})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add background for Berlin boundary
        if districts_gdf is not None:
            # Convert to same CRS
            districts_gdf_wgs84 = districts_gdf.to_crs(epsg=4326)
            
            # Plot district boundaries
            districts_gdf_wgs84.boundary.plot(ax=plt.gca(), color='gray', 
                                             linewidth=0.5, alpha=0.5)
        
        # Save figure if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network visualization saved to {output_path}")
        
        plt.show()
        return True
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return False

# Create visualization folder
VIZ_DIR = PROCESSED_DIR / 'visualizations'
VIZ_DIR.mkdir(exist_ok=True)

# Visualize network
visualize_network(final_stops_df, final_line_stops_df, final_line_df, 
                 output_path=VIZ_DIR / f'network_{YEAR}_{SIDE}.png')

# =============================================
# SUMMARY
# =============================================
logger.info("Processing complete")

print("\n" + "="*50)
print(f"SUMMARY: {YEAR} {SIDE.upper()} BERLIN PROCESSING")
print("="*50)
print(f"Lines processed: {len(final_line_df)}")
print(f"Stops processed: {len(final_stops_df)}")
print(f"Line-stop relationships processed: {len(final_line_stops_df)}")
print(f"Stops with location data: {final_stops_df['location'].notna().sum()} ({final_stops_df['location'].notna().sum()/len(final_stops_df)*100:.1f}%)")

if 'district' in final_stops_df.columns:
    print(f"Stops with district data: {final_stops_df['district'].notna().sum()} ({final_stops_df['district'].notna().sum()/len(final_stops_df)*100:.1f}%)")

print("\nFiles created:")
print(f"- {PROCESSED_DIR / f'lines_{YEAR}_{SIDE}.csv'}")
print(f"- {PROCESSED_DIR / f'stops_{YEAR}_{SIDE}.csv'}")
print(f"- {PROCESSED_DIR / f'line_stops_{YEAR}_{SIDE}.csv'}")
print(f"- {PROCESSED_DIR / 'existing_stations.csv'} (updated reference data)")
print(f"- {VIZ_DIR / f'network_{YEAR}_{SIDE}.png'} (network visualization)")

print("\nNext steps:")
print("1. Review and manually correct any stops missing location data")
print("2. Process East Berlin 1965 data using the same workflow")
print("3. Compare West and East Berlin networks using visualization tools")
print("4. Proceed to temporal analysis by processing adjacent years (1964, 1966)")
print("="*50)