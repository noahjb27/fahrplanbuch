import os

class Config:
    """Configuration settings for the application"""
    # Base directory of the application
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Directory one level above the base directory
    PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
    # Data directories - use absolute paths to ensure consistency
    DATA_DIR = os.path.join(PARENT_DIR, 'data')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MAPS_DIR = os.path.join(DATA_DIR, 'maps')
    
    # Make sure all directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)
    
    # Print paths for debugging
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"PARENT_DIR: {PARENT_DIR}")
    print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"MAPS_DIR: {MAPS_DIR}")
    
    # Database settings - could be SQLite for simplicity or whatever your existing DB is
    DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///' + os.path.join(BASE_DIR, 'berlin_transport.db'))
    
    # Secret key for sessions (generate a proper one for production)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # Default map settings
    DEFAULT_MAP_CENTER = [52.516667, 13.388889]  # Berlin center
    DEFAULT_ZOOM = 12
    
    # Export settings
    EXPORT_DIR = os.path.join(DATA_DIR, 'exports')
    os.makedirs(EXPORT_DIR, exist_ok=True)