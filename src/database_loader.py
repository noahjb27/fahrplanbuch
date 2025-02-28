"""
Improved Berlin Transport Loader with parallel processing and better batching.
"""

from ast import List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from time import sleep 
import pandas as pd

class BerlinTransportLoader:
    def __init__(self, uri: str, username: str, password: str, max_retries: int = 3):
        self.uri = uri
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.driver = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='transport_loader.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Establish connection to Neo4j"""
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        """Close the Neo4j connection"""
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute a database operation with retry logic"""
        retries = 0
        while retries < self.max_retries:
            try:
                self.connect()  # Ensure connection is established
                return func(*args, **kwargs)
            except (ServiceUnavailable, SessionExpired) as e:
                retries += 1
                self.logger.warning(f"Connection error (attempt {retries}/{self.max_retries}): {str(e)}")
                if retries == self.max_retries:
                    raise
                sleep(2 ** retries)  # Exponential backoff
                self.close()  # Close the current connection before retrying
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                raise

    def create_constraints(self):
        """Create necessary constraints for the database"""
        constraints = [
            "CREATE CONSTRAINT station_id IF NOT EXISTS FOR (s:Station) REQUIRE s.stop_id IS UNIQUE",
            "CREATE CONSTRAINT line_id IF NOT EXISTS FOR (l:Line) REQUIRE l.line_id IS UNIQUE",
            "CREATE CONSTRAINT district_name IF NOT EXISTS FOR (d:District) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT year_constraint IF NOT EXISTS FOR (y:Year) REQUIRE y.year IS UNIQUE",
            "CREATE CONSTRAINT postal_code IF NOT EXISTS FOR (p:PostalCode) REQUIRE p.code IS UNIQUE"
        ]
        
        def execute_constraints():
            with self.driver.session() as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        self.logger.info(f"Successfully created constraint: {constraint}")
                    except Exception as e:
                        self.logger.warning(f"Constraint creation failed: {str(e)}")
        
        self.execute_with_retry(execute_constraints)

    def create_schema(self, years: List[int]):
        """Create the complete schema with all node types and relationships"""
        def execute_schema():
            with self.driver.session() as session:
                years_query = """
                UNWIND $years AS year
                MERGE (y:Year {year: year})
                """
                session.run(years_query, years=years)
                self.logger.info(f"Successfully created Year nodes for years: {years}")
        
        self.execute_with_retry(execute_schema)

    def load_stations(self, stations_df: pd.DataFrame, batch_size: int = 100):
        """Load stations data with batch processing and updates"""
        station_query = """
        MATCH (y:Year {year: toInteger(substring(toString($stop_id), 0, 4))})
        MERGE (s:Station {stop_id: $stop_id})
        ON CREATE SET 
            s.name = $stop_name,
            s.type = $type,
            s.latitude = $latitude,
            s.longitude = $longitude,
            s.east_west = $east_west,
            s.description = $description
        ON MATCH SET
            s.name = CASE WHEN $stop_name IS NOT NULL THEN $stop_name ELSE s.name END,
            s.type = CASE WHEN $type IS NOT NULL THEN $type ELSE s.type END,
            s.latitude = CASE WHEN $latitude IS NOT NULL THEN $latitude ELSE s.latitude END,
            s.longitude = CASE WHEN $longitude IS NOT NULL THEN $longitude ELSE s.longitude END,
            s.east_west = CASE WHEN $east_west IS NOT NULL THEN $east_west ELSE s.east_west END,
            s.description = CASE WHEN $description IS NOT NULL THEN $description ELSE s.description END
            
        WITH s, y

        // Handle District relationship
        OPTIONAL MATCH (s)-[r1:IN_DISTRICT]->(:District)
        WITH s, y, CASE WHEN $bezirk IS NULL THEN [] ELSE [(s)-[:IN_DISTRICT]->(d:District {name: $bezirk}) | 1] END as new_district, r1
        FOREACH (x IN CASE WHEN r1 IS NOT NULL AND $bezirk IS NOT NULL AND r1.name <> $bezirk THEN [1] ELSE [] END |
            DELETE r1
        )
        FOREACH (x IN CASE WHEN $bezirk IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d:District {name: $bezirk})
            MERGE (s)-[:IN_DISTRICT]->(d)
        )

        // Handle Ortsteil relationship
        WITH s, y
        OPTIONAL MATCH (s)-[r2:IN_ORTSTEIL]->(:Ortsteil)
        WITH s, y, CASE WHEN $oteil IS NULL THEN [] ELSE [(s)-[:IN_ORTSTEIL]->(o:Ortsteil {name: $oteil}) | 1] END as new_ortsteil, r2
        FOREACH (x IN CASE WHEN r2 IS NOT NULL AND $oteil IS NOT NULL AND r2.name <> $oteil THEN [1] ELSE [] END |
            DELETE r2
        )
        FOREACH (x IN CASE WHEN $oteil IS NOT NULL THEN [1] ELSE [] END |
            MERGE (o:Ortsteil {name: $oteil})
            MERGE (s)-[:IN_ORTSTEIL]->(o)
        )

        // Handle PostalCode relationship
        WITH s, y
        OPTIONAL MATCH (s)-[r3:IN_POSTAL_CODE]->(:PostalCode)
        WITH s, y, CASE WHEN $plz IS NULL THEN [] ELSE [(s)-[:IN_POSTAL_CODE]->(p:PostalCode {code: $plz}) | 1] END as new_postal, r3
        FOREACH (x IN CASE WHEN r3 IS NOT NULL AND $plz IS NOT NULL AND r3.code <> $plz THEN [1] ELSE [] END |
            DELETE r3
        )
        FOREACH (x IN CASE WHEN $plz IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p:PostalCode {code: $plz})
            MERGE (s)-[:IN_POSTAL_CODE]->(p)
        )

        // Always ensure Year relationship
        WITH s, y
        MERGE (s)-[:IN_YEAR]->(y)
        """
        
        def process_batch(batch_df):
            with self.driver.session() as session:
                for _, row in batch_df.iterrows():
                    try:
                        # Parse location string to get lat/long
                        lat, lon = map(float, row['location'].strip('()').split(','))
                        
                        params = {
                            'stop_id': row['stop_id'],
                            'stop_name': row['stop_name'],
                            'type': row['type'],
                            'latitude': lat,
                            'longitude': lon,
                            'description': row['stop_description'],
                            'east_west': row['east-west'],
                            'bezirk': None if pd.isna(row['BEZIRK']) else row['BEZIRK'],
                            'oteil': None if pd.isna(row['OTEIL']) else row['OTEIL'],
                            'plz': None if pd.isna(row['plz']) else row['plz']
                        }
                        
                        session.run(station_query, params)
                        self.logger.debug(f"Successfully processed station: {row['stop_id']}")
                    except Exception as e:
                        self.logger.error(f"Error processing station {row['stop_id']}: {str(e)}")
                        raise

        # Process in batches
        for i in range(0, len(stations_df), batch_size):
            batch_df = stations_df.iloc[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} of {len(stations_df)//batch_size + 1}")
            self.execute_with_retry(process_batch, batch_df)

    def load_lines(self, lines_df: pd.DataFrame, batch_size: int = 100):
        """Load transport lines with batch processing and updates"""
        def process_batch(batch_df):
            with self.driver.session() as session:
                line_query = """
                MATCH (y:Year {year: $year})
                MERGE (l:Line {line_id: $line_id})
                ON CREATE SET 
                    l.name = $line_name,
                    l.type = $type,
                    l.east_west = $east_west,
                    l.frequency = $frequency,
                    l.capacity = $capacity,
                    l.length_time = $length_time,
                    l.length_km = $length_km,
                    l.profile = $profile
                ON MATCH SET
                    l.name = CASE WHEN $line_name IS NOT NULL THEN $line_name ELSE l.name END,
                    l.type = CASE WHEN $type IS NOT NULL THEN $type ELSE l.type END,
                    l.east_west = CASE WHEN $east_west IS NOT NULL THEN $east_west ELSE l.east_west END,
                    l.frequency = CASE WHEN $frequency IS NOT NULL THEN $frequency ELSE l.frequency END,
                    l.capacity = CASE WHEN $capacity IS NOT NULL THEN $capacity ELSE l.capacity END,
                    l.length_time = CASE WHEN $length_time IS NOT NULL THEN $length_time ELSE l.length_time END,
                    l.length_km = CASE WHEN $length_km IS NOT NULL THEN $length_km ELSE l.length_km END,
                    l.profile = CASE WHEN $profile IS NOT NULL THEN $profile ELSE l.profile END
                
                MERGE (l)-[:IN_YEAR]->(y)
                """
                
                for _, row in batch_df.iterrows():
                    try:
                        params = {
                            'line_id': row['line_id'],
                            'year': int(str(row['line_id'])[:4]),
                            'line_name': row['line_name'],
                            'type': row['type'],
                            'east_west': row['east_west'],
                            'frequency': row['Frequency'],
                            'capacity': row['capacity'],
                            'length_time': row['Length (time)'],
                            'length_km': row['Length (km)'],
                            'profile': row['profile']
                        }
                        session.run(line_query, params)
                        self.logger.debug(f"Successfully processed line: {row['line_id']}")
                    except Exception as e:
                        self.logger.error(f"Error processing line {row['line_id']}: {str(e)}")
                        raise

        # Process in batches by year
        for year in sorted(lines_df['line_id'].astype(str).str[:4].unique()):
            year_df = lines_df[lines_df['line_id'].astype(str).str[:4] == year]
            self.logger.info(f"Processing lines for year {year}")
            
            for i in range(0, len(year_df), batch_size):
                batch_df = year_df.iloc[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1} of {len(year_df)//batch_size + 1}")
                self.execute_with_retry(process_batch, batch_df)
                
            self.logger.info(f"Completed processing all lines for year {year}")

    def load_line_serves_relationships(self, line_stops_df: pd.DataFrame, batch_size: int = 50):
        """Create SERVES relationships between Lines and Stations"""
        def process_serves_batch(batch_df):
            with self.driver.session() as session:
                serves_query = """
                UNWIND $batch AS row
                MATCH (l:Line {line_id: row.line_id})
                MATCH (s:Station {stop_id: row.stop_id})
                MERGE (l)-[r:SERVES]->(s)
                SET r.stop_order = row.stop_order
                """
                try:
                    session.run(serves_query, {'batch': batch_df.to_dict('records')})
                    self.logger.debug(f"Successfully processed SERVES batch of size {len(batch_df)}")
                except Exception as e:
                    self.logger.error(f"Error processing SERVES batch: {str(e)}")
                    raise

        # Process in batches
        for i in range(0, len(line_stops_df), batch_size):
            batch_df = line_stops_df.iloc[i:i + batch_size]
            self.logger.info(f"Processing SERVES relationships batch {i//batch_size + 1}")
            self.execute_with_retry(process_serves_batch, batch_df)

    def load_connections(self, connections_df: pd.DataFrame, batch_size: int = 50):
        """Create CONNECTS_TO relationships between Stations using preprocessed connections"""
        def process_connections_batch(batch_df):
            with self.driver.session() as session:
                connection_query = """
                UNWIND $batch AS conn
                MATCH (s1:Station {stop_id: conn.from_stop_id})
                MATCH (s2:Station {stop_id: conn.to_stop_id})
                MERGE (s1)-[c:CONNECTS_TO]->(s2)
                SET 
                    c.distance_meters = conn.distance_meters,
                    c.time_minutes = conn.time_minutes,
                    c.transport_type = conn.transport_type,
                    c.line_ids = conn.line_id,
                    c.line_names = conn.line_name,
                    c.capacities = conn.capacity,
                    c.frequencies = conn.frequency,
                    c.hourly_capacity = conn.hourly_capacity,
                    c.hourly_services = conn.hourly_services
                """
                try:
                    session.run(connection_query, {'batch': batch_df.to_dict('records')})
                    self.logger.debug(f"Successfully processed CONNECTS_TO batch of size {len(batch_df)}")
                except Exception as e:
                    self.logger.error(f"Error processing CONNECTS_TO batch: {str(e)}")
                    raise

        # Process in batches
        for i in range(0, len(connections_df), batch_size):
            batch_df = connections_df.iloc[i:i + batch_size]
            self.logger.info(f"Processing CONNECTS_TO relationships batch {i//batch_size + 1}")
            self.execute_with_retry(process_connections_batch, batch_df)
    
    def verify_data(self):
        """Verify that all data was loaded correctly"""
        def execute_verification():
            with self.driver.session() as session:
                # Check all node types
                verification_queries = {
                    'stations': "MATCH (s:Station) RETURN count(s) as count",
                    'lines': "MATCH (l:Line) RETURN count(l) as count",
                    'years': "MATCH (y:Year) RETURN count(y) as count",
                    'districts': "MATCH (d:District) RETURN count(d) as count",
                    'postcodes': "MATCH (p:PostalCode) RETURN count(p) as count",
                    'ortsteile': "MATCH (o:Ortsteil) RETURN count(o) as count"
                }
                
                # Check relationships
                relationship_queries = {
                    'serves': "MATCH ()-[r:SERVES]->() RETURN count(r) as count",
                    'connects': "MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) as count",
                    'in_year': "MATCH ()-[r:IN_YEAR]->() RETURN count(r) as count"
                }
                
                results = {}
                for name, query in {**verification_queries, **relationship_queries}.items():
                    count = session.run(query).single()["count"]
                    results[name] = count
                    self.logger.info(f"{name}: {count}")
                
                # Log summary by year
                year_summary_query = """
                MATCH (y:Year)
                WITH y
                OPTIONAL MATCH (s:Station)-[:IN_YEAR]->(y)
                WITH y, count(s) as station_count
                OPTIONAL MATCH (l:Line)-[:IN_YEAR]->(y)
                WITH y, station_count, count(l) as line_count
                RETURN y.year as year, station_count, line_count
                ORDER BY y.year
                """
                
                year_results = session.run(year_summary_query)
                for record in year_results:
                    self.logger.info(
                        f"Year {record['year']}: {record['station_count']} stations, "
                        f"{record['line_count']} lines"
                    )
                
                # Check for potential issues
                issues = []
                if results['stations'] == 0:
                    issues.append("No stations found")
                if results['lines'] == 0:
                    issues.append("No lines found")
                if results['serves'] == 0:
                    issues.append("No SERVES relationships found")
                if results['connects'] == 0:
                    issues.append("No CONNECTS_TO relationships found")
                    
                if issues:
                    for issue in issues:
                        self.logger.warning(issue)
                    return False
                return True

        return self.execute_with_retry(execute_verification)

class BerlinTransportLoaderImproved(BerlinTransportLoader):
    def __init__(self, uri: str, username: str, password: str, max_retries: int = 3):
        super().__init__(uri, username, password, max_retries)
        self.batch_stats = defaultdict(int)
        
    def load_multiple_years(self, years: List[int], data_dir: Path):
        """
        Load data for multiple years with progress tracking
        and parallel preprocessing.
        """
        total_progress = len(years) * 4  # stations, lines, serves, connections
        current_progress = 0
        
        def update_progress():
            nonlocal current_progress
            current_progress += 1
            self.logger.info(f"Overall progress: {current_progress}/{total_progress}")
            
        # Create schema first
        self.create_schema(years)
        
        # Process each year
        for year in years:
            self.logger.info(f"\nProcessing year {year}")
            
            # Load data files
            try:
                stations_df = pd.read_csv(data_dir / f'stops_{year}_enriched.csv')
                lines_df = pd.read_csv(data_dir / f'lines_{year}_enriched.csv')
                line_stops_df = pd.read_csv(data_dir / f'line_stops_{year}.csv')
                connections_df = pd.read_csv(data_dir / f'connections_{year}.csv')
                
                # Preprocess in parallel
                with ThreadPoolExecutor() as executor:
                    future_stations = executor.submit(self._preprocess_stations, stations_df)
                    future_lines = executor.submit(self._preprocess_lines, lines_df)
                    
                    # Wait for preprocessing to complete
                    stations_df = future_stations.result()
                    lines_df = future_lines.result()
                
                # Load data with progress tracking
                self.load_stations(stations_df)
                update_progress()
                
                self.load_lines(lines_df)
                update_progress()
                
                self.load_line_serves_relationships(line_stops_df)
                update_progress()
                
                self.load_connections(connections_df)
                update_progress()
                
            except FileNotFoundError as e:
                self.logger.warning(f"Missing file for year {year}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
                raise
                
        # Verify final data
        if self.verify_data():
            self.logger.info("Data loading completed successfully")
        else:
            self.logger.error("Data verification failed")
            
    def _preprocess_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess stations data for better loading performance."""
        # Convert location string to lat/lon
        df[['latitude', 'longitude']] = df['location'].str.split(',', expand=True).astype(float)
        
        # Clean and standardize data
        df = df.fillna('')
        for col in ['stop_name', 'type', 'east_west']:
            df[col] = df[col].astype(str).str.strip()
            
        return df
        
    def _preprocess_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess lines data for better loading performance."""
        # Convert numeric columns
        numeric_cols = ['Frequency', 'capacity', 'Length (time)', 'Length (km)']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Clean string columns
        for col in ['line_name', 'type', 'east_west', 'profile']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                
        return df
        
    def load_stations(self, stations_df: pd.DataFrame, batch_size: int = 1000):
        """Improved station loading with larger batches and better error handling."""
        def create_station_params(row):
            try:
                return {
                    'stop_id': row['stop_id'],
                    'stop_name': row['stop_name'],
                    'type': row['type'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'east_west': row['east_west'],
                    'bezirk': None if pd.isna(row.get('BEZIRK')) else row['BEZIRK'],
                    'oteil': None if pd.isna(row.get('OTEIL')) else row['OTEIL'],
                    'plz': None if pd.isna(row.get('plz')) else row['plz']
                }
            except Exception as e:
                self.logger.error(f"Error creating params for station {row['stop_id']}: {e}")
                raise
                
        def process_station_batch(batch_df):
            with self.driver.session() as session:
                params_list = [create_station_params(row) for _, row in batch_df.iterrows()]
                try:
                    session.run(
                        "UNWIND $params AS param " + self.STATION_QUERY,
                        params={'params': params_list}
                    )
                    self.batch_stats['stations'] += len(batch_df)
                except Exception as e:
                    self.logger.error(f"Error processing station batch: {e}")
                    raise
                    
        super().load_stations(stations_df, batch_size)