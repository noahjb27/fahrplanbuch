# db_station_matcher.py

import pandas as pd
import logging
from neo4j import GraphDatabase
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DBStationMatcher:
    """Match stations with Neo4j database records."""
    
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="password"):
        """
        Initialize matcher with Neo4j database connection.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def add_location_data(self, stops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add location data to stops dataframe from database.
        
        Args:
            stops_df: DataFrame containing stops to process
            
        Returns:
            DataFrame with added location data where matches found
        """
        result_df = stops_df.copy()
        match_count = 0
        total_stops = len(result_df)
        
        # Initialize location and identifier columns with None
        result_df['location'] = None
        result_df['identifier'] = None
        
        for idx, row in result_df.iterrows():
            # Extract year from stop_id (assuming format like '19650_east')
            if '_' in row['stop_id']:
                year = row['stop_id'].split('_')[0][:4]
            else:
                year = row['stop_id'][:4]
                
            try:
                year = int(year)
            except ValueError:
                year = 0
                
            # Find matches in database
            matches = self._find_matching_stations(
                station_name=row['stop_name'],
                station_type=row['type'],
                line_name=row['line_name'],
                current_year=year
            )
            
            if matches:
                best_match = matches[0]  # Use the most recent match
                # Only count as match if location exists
                if 'latitude' in best_match and 'longitude' in best_match:
                    match_count += 1
                    lat, lng = best_match['latitude'], best_match['longitude']
                    result_df.loc[idx, 'location'] = f"{lat},{lng}"
                    result_df.loc[idx, 'identifier'] = best_match.get('wikidata_id', '')
                    
                    logger.debug(f"Matched station {row['stop_name']} with DB record from year {best_match.get('year')}")
                else:
                    logger.warning(f"Found match for station {row['stop_name']} but no location data available")
            else:
                logger.info(f"No match found for station: {row['stop_name']}")
                
        logger.info(f"Successfully matched {match_count} out of {total_stops} stations")
        
        return result_df
    
    def _find_matching_stations(self, station_name: str, station_type: str, 
                              line_name: str, current_year: int) -> List[Dict]:
        """
        Find stations matching name/type/line from previous years.
        
        Args:
            station_name: Name of the station
            station_type: Type of station (u-bahn, s-bahn, etc.)
            line_name: Name of the line
            current_year: Current year being processed
            
        Returns:
            List of matching stations from the database
        """
        # First try exact match with name, type, and line
        with self.driver.session() as session:
            query = """
            MATCH (s:Station {name: $name, type: $type})
            WHERE exists((s)-[:SERVES]-(:Line {name: $line_name}))
            WITH s
            MATCH (s)-[:IN_YEAR]->(y:Year)
            WHERE toInteger(y.year) < $year
            RETURN s.stop_id as stop_id, s.name as name, s.type as type,
                   s.latitude as latitude, s.longitude as longitude, 
                   s.wikidata_id as wikidata_id, y.year as year
            ORDER BY y.year DESC
            LIMIT 1
            """
            
            result = session.run(
                query, 
                name=station_name,
                type=station_type,
                line_name=line_name,
                year=current_year
            )
            exact_matches = result.data()
            
            if exact_matches:
                return exact_matches
            
            # If no exact match, try just name and type
            query = """
            MATCH (s:Station {name: $name, type: $type})
            WITH s
            MATCH (s)-[:IN_YEAR]->(y:Year)
            WHERE toInteger(y.year) < $year
            RETURN s.stop_id as stop_id, s.name as name, s.type as type,
                   s.latitude as latitude, s.longitude as longitude, 
                   s.wikidata_id as wikidata_id, y.year as year
            ORDER BY y.year DESC
            LIMIT 5
            """
            
            result = session.run(
                query, 
                name=station_name,
                type=station_type,
                year=current_year
            )
            return result.data()