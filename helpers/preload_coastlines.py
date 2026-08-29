#!/usr/bin/env python3
"""
Bulk pre-download coastline data for a given bounding box.
Downloads from Overpass API (with multiple endpoint failover) and saves to local NPZ/JSON cache.
Run once, then use offline in pipeline.
"""
import os
import json
import logging
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("coastline_preload")

# Multiple Overpass endpoints for failover
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter", 
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass-api.mandel.io/api/interpreter",
    "https://overpass-api.tokyo.cloud/api/interpreter",
]

def fetch_tile(endpoint, min_lat, min_lon, max_lat, max_lon, timeout):
    """Fetch coastline data for a single tile from a specific endpoint."""
    query = f"""
    [out:json][timeout:{int(timeout)}];
    (way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon}););
    out geom;
    """
    response = requests.get(endpoint, params={'data': query},
                            headers={'User-Agent': 'ML-Pipeline/1.0'}, timeout=timeout)
    response.raise_for_status()
    elements = response.json().get('elements', [])
    return [[(node['lon'], node['lat']) for node in el['geometry']]
            for el in elements if 'geometry' in el]

def fetch_tile_with_failover(min_lat, min_lon, max_lat, max_lon, timeout=30):
    """Try all endpoints until one succeeds."""
    for i, endpoint in enumerate(OVERPASS_ENDPOINTS):
        try:
            return fetch_tile(endpoint, min_lat, min_lon, max_lat, max_lon, timeout)
        except requests.RequestException as exc:
            log.debug("Endpoint %d (%s) failed: %s", i + 1, endpoint, exc)
            continue
    raise RuntimeError("All Overpass endpoints failed")

def preload_coastlines(min_lat, min_lon, max_lat, max_lon, 
                       tile_deg=0.1, timeout=30, max_workers=4,
                       output_dir="coastline_preload"):
    """
    Download coastlines for entire bbox in tiles.
    
    Args:
        min_lat, min_lon, max_lat, max_lon: Bounding box
        tile_deg: Tile size in degrees (0.1° ≈ 11km)
        timeout: Seconds per request
        max_workers: Parallel downloads
        output_dir: Where to save cache
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate tile grid
    lat_edges = np.arange(min_lat, max_lat, tile_deg)
    lon_edges = np.arange(min_lon, max_lon, tile_deg)
    
    # Ensure we cover the full extent
    if lat_edges[-1] < max_lat:
        lat_edges = np.append(lat_edges, max_lat)
    if lon_edges[-1] < max_lon:
        lon_edges = np.append(lon_edges, max_lon)
    
    tiles = []
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            tiles.append((
                lat_edges[i], lon_edges[j],
                lat_edges[i + 1], lon_edges[j + 1]
            ))
    
    log.info("Preloading coastlines: %d tiles of %.2f° each", len(tiles), tile_deg)
    log.info("BBox: lat [%.4f, %.4f], lon [%.4f, %.4f]", min_lat, max_lat, min_lon, max_lon)
    
    # Download in parallel
    all_coastlines = []
    failed_tiles = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {
            executor.submit(fetch_tile_with_failover, t[0], t[1], t[2], t[3], timeout): t
            for t in tiles
        }
        
        for future in as_completed(future_to_tile):
            tile = future_to_tile[future]
            try:
                coastlines = future.result()
                if coastlines:
                    all_coastlines.extend(coastlines)
                    log.info("Tile (%.3f,%.3f)-(%.3f,%.3f): %d segments", 
                             tile[0], tile[1], tile[2], tile[3], len(coastlines))
                else:
                    log.info("Tile (%.3f,%.3f)-(%.3f,%.3f): no coastline data", 
                             tile[0], tile[1], tile[2], tile[3])
            except Exception as exc:
                log.warning("Tile (%.3f,%.3f)-(%.3f,%.3f) failed: %s", 
                            tile[0], tile[1], tile[2], tile[3], exc)
                failed_tiles.append(tile)
    
    # Save combined data
    output_file = os.path.join(output_dir, f"coastlines_{min_lat:.2f}_{min_lon:.2f}_{max_lat:.2f}_{max_lon:.2f}.json")
    with open(output_file, 'w') as f:
        json.dump(all_coastlines, f)
    
    log.info("Saved %d coastline segments to %s", len(all_coastlines), output_file)
    if failed_tiles:
        log.warning("%d tiles failed: %s", len(failed_tiles), failed_tiles)
    
    return all_coastlines, output_file

def load_preloaded_coastlines(cache_file):
    """Load preloaded coastlines from file."""
    with open(cache_file) as f:
        return json.load(f)

def coastlines_near_preloaded(all_coastlines, min_lat, min_lon, max_lat, max_lon):
    """Filter preloaded coastlines to bbox (same interface as coastlines_near)."""
    out = []
    for line in all_coastlines:
        lons = [p[0] for p in line]
        lats = [p[1] for p in line]
        if max(lons) < min_lon or min(lons) > max_lon or max(lats) < min_lat or min(lats) > max_lat:
            continue
        out.append(line)
    return out


if __name__ == "__main__":
    # Houston/Galveston area from your dataset
    MIN_LAT, MAX_LAT = 29.68, 29.78
    MIN_LON, MAX_LON = -95.29, -94.98
    
    coastlines, cache_file = preload_coastlines(
        MIN_LAT, MIN_LON, MAX_LAT, MAX_LON,
        tile_deg=0.1,      # 0.1° tiles (11km)
        timeout=30,
        max_workers=4,     # Parallel downloads
        output_dir="coastline_preload"
    )
    print(f"Done! Cache file: {cache_file}")
    print(f"Total segments: {len(coastlines)}")