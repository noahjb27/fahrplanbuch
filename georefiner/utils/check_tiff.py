"""
Utility script to check if a GeoTIFF file is properly georeferenced.
Run this from the command line to check your TIF files:

python utils/check_tiff.py data/maps/your_file.tif
"""

import sys
import os
import rasterio
from rasterio.errors import RasterioIOError

def check_tiff(filepath):
    """Check if a TIFF file is properly georeferenced and can be read by rasterio"""
    print(f"Checking file: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return False
    
    try:
        with rasterio.open(filepath) as src:
            print("File opened successfully")
            print(f"Driver: {src.driver}")
            print(f"Width: {src.width}, Height: {src.height}")
            print(f"Number of bands: {src.count}")
            print(f"CRS: {src.crs}")
            
            if not src.crs:
                print("Warning: No coordinate reference system (CRS) defined")
                print("This file is not properly georeferenced")
                print("\nSuggestion: Add georeferencing with GDAL:")
                print(f"gdal_translate -a_srs EPSG:4326 {filepath} {filepath.replace('.tif', '_georeferenced.tif')}")
                return False
            
            print(f"Bounds: {src.bounds}")
            print(f"Resolution: {src.res}")
            print(f"Transform: {src.transform}")
            
            # Read a small sample to ensure data can be accessed
            try:
                window = ((0, min(10, src.height)), (0, min(10, src.width)))
                data = src.read(1, window=window)
                print(f"Successfully read data sample of shape: {data.shape}")
            except Exception as e:
                print(f"Error reading data: {e}")
                return False
            
            return True
            
    except RasterioIOError as e:
        print(f"Error opening file: {e}")
        print("\nPossible causes:")
        print("1. File is not a valid GeoTIFF")
        print("2. File is corrupted")
        print("3. Missing required GDAL drivers")
        print("\nSuggestion: Install GDAL and dependencies:")
        print("sudo apt-get update")
        print("sudo apt-get install python3-gdal gdal-bin libgdal-dev")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_tiff.py <path_to_tiff_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = check_tiff(file_path)
    
    if success:
        print("\nSummary: File appears to be a valid, georeferenced GeoTIFF")
    else:
        print("\nSummary: Issues were found with this file")
        sys.exit(1)