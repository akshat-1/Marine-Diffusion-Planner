#!/usr/bin/env python3
"""
Offline coastline data loader using Natural Earth or GSHHG shapefiles.
No API calls needed - download once, use forever.
"""
import os
import json
import logging
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("offline_coastline")

# Natural Earth coastline URLs (download once) - using CDN for direct access
NATURAL_EARTH_URLS = {
    "10m": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
    "50m": "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip",
    "110m": "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip",
}

# GSHHG (higher resolution) - alternative
GSHHG_URL = "https://www.ngdc.noaa.gov/mgg/shorelines/gshhs_latest.zip"

def download_and_extract(url, output_dir):
    """Download and extract a zip file."""
    import urllib.request
    import zipfile
    import tempfile
    
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, os.path.basename(url))
    
    if not os.path.exists(zip_path):
        log.info("Downloading %s...", url)
        urllib.request.urlretrieve(url, zip_path)
        log.info("Downloaded to %s", zip_path)
    
    # Extract
    extract_dir = os.path.join(output_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find shapefile
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.shp') and 'coastline' in f.lower():
                return os.path.join(root, f)
    return None

def shapefile_to_coastlines(shapefile_path, min_lat, min_lon, max_lat, max_lon):
    """Convert shapefile to coastline format: list of [(lon, lat), ...]"""
    if not GEOPANDAS_AVAILABLE:
        raise RuntimeError("geopandas required. Install: pip install geopandas")
    
    log.info("Loading shapefile: %s", shapefile_path)
    gdf = gpd.read_file(shapefile_path)
    
    # Filter to bbox using shapely box
    from shapely.geometry import box
    bbox_poly = box(min_lon, min_lat, max_lon, max_lat)
    
    gdf = gdf[gdf.intersects(bbox_poly)]
    
    coastlines = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        # Clip geometry to bbox to get only the relevant portion
        clipped = geom.intersection(bbox_poly)
        
        if clipped.is_empty:
            continue
            
        if clipped.geom_type == 'LineString':
            coords = list(clipped.coords)
            if len(coords) >= 2:
                coastlines.append(coords)
        elif clipped.geom_type == 'MultiLineString':
            for line in clipped.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    coastlines.append(coords)
        elif clipped.geom_type == 'GeometryCollection':
            for part in clipped.geoms:
                if part.geom_type == 'LineString':
                    coords = list(part.coords)
                    if len(coords) >= 2:
                        coastlines.append(coords)
    
    log.info("Extracted %d coastline segments", len(coastlines))
    return coastlines

def save_coastlines_json(coastlines, output_path):
    """Save coastlines to JSON cache file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coastlines, f)
    log.info("Saved %d segments to %s", len(coastlines), output_path)

def load_coastlines_json(cache_path):
    """Load coastlines from JSON cache."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        return json.load(f)

def coastlines_near(all_coastlines, min_lat, min_lon, max_lat, max_lon):
    """Filter coastlines to bbox (compatible with pipeline)."""
    out = []
    for line in all_coastlines:
        lons = [p[0] for p in line]
        lats = [p[1] for p in line]
        if max(lons) < min_lon or min(lons) > max_lon or max(lats) < min_lat or min(lats) > max_lat:
            continue
        out.append(line)
    return out


def prepare_offline_coastlines(resolution="10m", bbox=None, cache_dir="/run/media/akshat/Akshat_USB/coastline_offline"):
    """
    Main function to prepare offline coastline data.
    
    Args:
        resolution: "10m" (high), "50m" (medium), "110m" (low)
        bbox: (min_lat, min_lon, max_lat, max_lon) or None for full dataset
        cache_dir: Where to store downloaded/processed data (defaults to USB)
    """
    if bbox is None:
        # Your Houston/Galveston dataset bounds
        bbox = (29.68, -95.29, 29.78, -94.98)
    
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Check cache first
    cache_file = os.path.join(cache_dir, f"coastlines_{resolution}_{min_lat:.2f}_{min_lon:.2f}_{max_lat:.2f}_{max_lon:.2f}.json")
    
    cached = load_coastlines_json(cache_file)
    if cached is not None:
        log.info("Using cached coastlines: %d segments", len(cached))
        return cached
    
    # Download Natural Earth data
    url = NATURAL_EARTH_URLS.get(resolution, NATURAL_EARTH_URLS["10m"])
    shapefile = download_and_extract(url, os.path.join(cache_dir, "natural_earth"))
    
    if shapefile is None:
        raise RuntimeError("Could not find coastline shapefile in download")
    
    # Convert to our format
    coastlines = shapefile_to_coastlines(shapefile, min_lat, min_lon, max_lat, max_lon)
    
    # Save cache
    save_coastlines_json(coastlines, cache_file)
    
    return coastlines


if __name__ == "__main__":
    import sys
    
    # Check dependencies
    if not GEOPANDAS_AVAILABLE:
        print("geopandas not installed. Install with:")
        print("  pip install geopandas")
        sys.exit(1)
    
    # Houston/Galveston area
    bbox = (29.68, -95.29, 29.78, -94.98)
    
    # Use 10m (highest resolution)
    coastlines = prepare_offline_coastlines(resolution="10m", bbox=bbox)
    print(f"Prepared {len(coastlines)} coastline segments for offline use")