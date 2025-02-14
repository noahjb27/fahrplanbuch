import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import logging

from shapely import Point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataEnricher:
    def __init__(self, year: int, side: str):
        self.year = year
        self.side = side
        self.data_dir = Path('../data')
        
    def load_and_combine_data(self):
        """Load and combine matched and refined data."""
        # Load matched stations
        matched_path = self.data_dir / 'interim' / 'stops_matched' / f'stops_{self.year}_{self.side}.csv'
        matched_stops = pd.read_csv(matched_path)
        
        # Load refined stations
        refined_path = self.data_dir / 'interim' / 'stops_refined' / f'stops_{self.year}_{self.side}_refined.csv'
        refined_stops = pd.read_csv(refined_path)
        
        # Combine data
        final_stops = matched_stops.copy()
        mask = final_stops['location'].isna()
        final_stops.loc[mask] = refined_stops[mask]
        
        self.stops_df = final_stops
        logger.info(f"Loaded {len(self.stops_df)} total stations")
        
    def add_administrative_data(self):
        """Add administrative boundary information."""
        # Load GeoJSON data
        districts_gdf = gpd.read_file("../data-external/lor_ortsteile.geojson")
        
        # Convert stations to GeoDataFrame
        geometry = self.stops_df['location'].apply(lambda x: 
            Point(*map(float, x.split(','))))
        stops_gdf = gpd.GeoDataFrame(self.stops_df, geometry=geometry)
        
        # Perform spatial join
        stops_with_admin = gpd.sjoin(
            stops_gdf, 
            districts_gdf[['geometry', 'BEZIRK', 'OTEIL']], 
            how="left", 
            predicate='within'
        )
        
        self.stops_df = stops_with_admin
        logger.info("Added administrative data")
        
    def add_east_west_classification(self):
        """Add East/West classification based on location."""
        # Load West Berlin districts
        with open("../data-external/West-Berlin-Ortsteile.json", "r") as f:
            west_berlin = json.load(f)["West_Berlin"]
            
        # Classify stations
        self.stops_df['east_west'] = self.stops_df['OTEIL'].apply(
            lambda x: 'west' if x in west_berlin else 'east'
        )
        
        logger.info("Added East/West classification")
        
    def enrich_transport_data(self):
        """Add transport-specific enrichment."""
        # Calculate transfer points
        transfer_points = self.stops_df.groupby('stop_name').agg({
            'type': 'nunique',
            'in_lines': 'count'
        }).reset_index()
        
        # Add transfer point flag
        self.stops_df = self.stops_df.merge(
            transfer_points,
            on='stop_name',
            suffixes=('', '_count')
        )
        
        self.stops_df['is_transfer'] = (
            self.stops_df['type_count'] > 1) | (self.stops_df['in_lines_count'] > 1
        )
        
        logger.info("Added transport enrichment")
        
    def prepare_for_neo4j(self):
        """Prepare data for Neo4j import."""
        # Add required fields
        self.stops_df['created_at'] = pd.Timestamp.now()
        self.stops_df['source'] = f'fahrplanbuch_{self.year}'
        
        # Clean and validate data
        self.stops_df = self.stops_df.fillna('')
        
        return self.stops_df