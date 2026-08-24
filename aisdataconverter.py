import os
import json
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import LineString
from scipy.spatial import cKDTree

from commonocean.scenario.scenario import Scenario, Tag
from commonocean.scenario.obstacle import DynamicObstacle, StaticObstacle, ObstacleType
from commonocean.scenario.waters import Shallow
from commonocean.scenario.trajectory import Trajectory
from commonocean.prediction.prediction import TrajectoryPrediction
from commonocean.common.file_writer import CommonOceanFileWriter, OverwriteExistingFile
from commonocean.scenario.state import InitialState, TFState
from commonroad.geometry.shape import Rectangle, Polygon as CRPolygon
from commonroad.planning.planning_problem import PlanningProblemSet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ais_pipeline")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MIN_DISPLACEMENT_METERS = 50.0   # A ship MUST physically travel at least 50 meters from start to finish
DT_SECONDS = 10.0
WINDOW_MINUTES = 20
WINDOW_STEP_MINUTES = 5
MAX_GAP_BINS = 45                 # Bridges 7.5-minute radio gaps
MIN_TRAJECTORY_LEN = 61
TARGET_FRAMES = 61          
MAX_SCENE_RADIUS_M = 8000.0       
MIN_SPEED_KNOTS_FOR_ANCHORED = 0.5
LOCAL_BOX_DEG = 0.05
FLEET_OVERLAP_THRESHOLD = 0.4
MAX_SHIPS_PER_SCENE = 12
DEFAULT_LENGTH_M = 150.0
DEFAULT_WIDTH_M = 25.0
DEFAULT_DRAFT_M = 5.0
SAFE_CPA_METERS = 15.0

MAX_SCENARIOS_PER_GRID_CELL = 100

# --- MANEUVER FILTER TUNABLES ---
MIN_TURN_DEG_PER_SCENE = 10.0     
MIN_DYNAMIC_SPEED_KNOTS = 2.0     

# --- LARGE SHIP / EGO PRIORITIZATION ---
LARGE_SHIP_LENGTH_M = 100.0           # Cargo/tanker threshold (lowered from 120)
CARGO_SPEED_KNOTS = 3.0               # Realistic cargo transit speed (lowered from 5)
TARGET_LARGE_MOVERS_PER_SCENE = 1     # Minimum large movers per scenario (lowered from 2)
EGO_LARGE_TARGET_RATIO = 0.6          # Target fraction of scenarios with large ego
# --- MIXED SAMPLING ---
SAMPLING_MODE = "mixed"               # "strict" | "relaxed" | "mixed"
STRICT_SAMPLING_RATIO = 0.80          # In mixed mode, 80% strict / 20% relaxed for diversity without overfiting
# ------------------------------------

# --- NEW: Per-scenario coastline config ---
COASTLINE_CACHE_DIR = "coastline_cache"
COASTLINE_PER_SCENARIO = True          # Fetch per-scenario (fast, reliable) vs one-time global
COASTLINE_BBOX_DEG = 0.1               # BBox around anchor (0.1° ≈ 11km)
COASTLINE_TIMEOUT = 30                 # Seconds per request (increased from 15)
COASTLINE_MAX_RETRIES = 5              # Max retries (increased from 2)

# --- OFFLINE COASTLINE CONFIG ---
OFFLINE_COASTLINE_DIR = "/run/media/akshat/Akshat_USB/coastline_offline"  # Path to preloaded coastline data
OFFLINE_COASTLINE_FILE = "coastlines_10m_29.68_-95.29_29.78_-94.98.json"  # Specific file for Houston area
USE_OFFLINE_COASTLINES = True          # Set True to use preloaded data, False for API

BATHYMETRY_NPZ_PATH = "bathymetry/Singaporebathymetry.npz"
SHALLOW_SAFETY_MARGIN_M = 2.0

EXCLUSION_ZONES = []

# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------
def vessel_type_to_obstacle_type(code, sog_ms_mean):
    if sog_ms_mean is not None and sog_ms_mean < (MIN_SPEED_KNOTS_FOR_ANCHORED * 0.514444):
        return ObstacleType.ANCHOREDVESSEL
    try:
        code = int(code)
    except (TypeError, ValueError):
        return ObstacleType.MOTORVESSEL
    if code == 30: return ObstacleType.FISHINGVESSEL
    if code == 36: return ObstacleType.SAILINGVESSEL
    if code == 35: return ObstacleType.MILITARYVESSEL
    if 70 <= code <= 89: return ObstacleType.CARGOSHIP
    return ObstacleType.MOTORVESSEL

def is_cargo_or_tanker(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    return 70 <= code <= 89

def apply_exclusion_zones(df):
    if not EXCLUSION_ZONES:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for (min_lat, min_lon, max_lat, max_lon) in EXCLUSION_ZONES:
        in_zone = (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) & \
                  (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
        mask |= in_zone
    dropped = mask.sum()
    if dropped > 0:
        log.info("Excluding %d rows (%.2f%%) inside EXCLUSION_ZONES before anchor selection.", dropped, (dropped/len(df))*100)
    return df[~mask].reset_index(drop=True)

def is_in_exclusion_zone(lat, lon):
    for (min_lat, min_lon, max_lat, max_lon) in EXCLUSION_ZONES:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return True
    return False

# --- Multiple Overpass API endpoints for failover ---
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",           # Main (Germany)
    "https://overpass.kumi.systems/api/interpreter",     # Kumi Systems (Japan)
    "https://overpass.openstreetmap.ru/api/interpreter", # Russia
    "https://overpass-api.mandel.io/api/interpreter",    # Mandel (UK)
    "https://overpass-api.tokyo.cloud/api/interpreter",  # Tokyo
]

def _fetch_overpass_tile(min_lat, min_lon, max_lat, max_lon, timeout):
    overpass_query = f"""
    [out:json][timeout:{int(timeout)}];
    (way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon}););
    out geom;
    """
    
    # Try each endpoint in order until one succeeds
    last_exception = None
    for i, endpoint in enumerate(OVERPASS_ENDPOINTS):
        try:
            response = requests.get(endpoint, params={'data': overpass_query},
                                     headers={'User-Agent': 'ML-Pipeline/1.0'}, timeout=timeout)
            response.raise_for_status()
            elements = response.json().get('elements', [])
            if i > 0:
                log.info("Coastline fetch succeeded via fallback endpoint #%d: %s", i + 1, endpoint)
            return [[(node['lon'], node['lat']) for node in el['geometry']]
                    for el in elements if 'geometry' in el]
        except requests.RequestException as exc:
            last_exception = exc
            log.debug("Overpass endpoint %d (%s) failed: %s", i + 1, endpoint, exc)
            continue
    
    # All endpoints failed
    raise last_exception or RuntimeError("All Overpass endpoints failed")

def coastlines_near(all_coastlines, min_lat, min_lon, max_lat, max_lon):
    out = []
    for line in all_coastlines:
        lons = [p[0] for p in line]
        lats = [p[1] for p in line]
        if max(lons) < min_lon or min(lons) > max_lon or max(lats) < min_lat or min(lats) > max_lat:
            continue
        out.append(line)
    return out

# --- OFFLINE: Load preloaded coastline data from USB ---
_offline_coastlines_cache = None

def load_offline_coastlines():
    """Load preloaded coastline data from USB drive (runs once, caches in memory)."""
    global _offline_coastlines_cache
    if _offline_coastlines_cache is not None:
        return _offline_coastlines_cache
    
    offline_path = os.path.join(OFFLINE_COASTLINE_DIR, OFFLINE_COASTLINE_FILE)
    if not os.path.exists(offline_path):
        log.warning("Offline coastline file not found: %s", offline_path)
        _offline_coastlines_cache = []
        return []
    
    try:
        with open(offline_path) as f:
            _offline_coastlines_cache = json.load(f)
        log.info("Loaded %d offline coastline segments from %s", len(_offline_coastlines_cache), offline_path)
        return _offline_coastlines_cache
    except Exception as exc:
        log.error("Failed to load offline coastlines: %s", exc)
        _offline_coastlines_cache = []
        return []

# --- NEW: Fast, reliable per-scenario coastline fetching ---
def fetch_coastlines_for_scenario(anchor_lat, anchor_lon, bbox_deg=COASTLINE_BBOX_DEG,
                                   cache_dir=COASTLINE_CACHE_DIR, timeout=COASTLINE_TIMEOUT,
                                   max_retries=COASTLINE_MAX_RETRIES, stats=None):
    """
    Fetch coastlines for a single scenario's bounding box.
    Uses offline preloaded data if USE_OFFLINE_COASTLINES=True, otherwise Overpass API.
    Returns [] on failure (never crashes pipeline).
    """
    
    # --- OFFLINE MODE: Use preloaded coastline data ---
    if USE_OFFLINE_COASTLINES:
        all_offline = load_offline_coastlines()
        if all_offline:
            # Filter offline data to scenario bbox
            min_lat = anchor_lat - bbox_deg
            min_lon = anchor_lon - bbox_deg
            max_lat = anchor_lat + bbox_deg
            max_lon = anchor_lon + bbox_deg
            coastlines = coastlines_near(all_offline, min_lat, min_lon, max_lat, max_lon)
            if stats is not None:
                stats['coastlines_cached'] += 1
            return coastlines
        else:
            log.warning("Offline coastline data empty, falling back to API")
    
    # --- ONLINE MODE: Original API fetching with caching ---
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache key: rounded anchor coordinates (deterministic, ~1km resolution)
    cache_key = (round(anchor_lat, 2), round(anchor_lon, 2))
    cache_file = os.path.join(cache_dir, f"scenario_{cache_key[0]:.2f}_{cache_key[1]:.2f}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                if stats is not None:
                    stats['coastlines_cached'] += 1
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # Corrupted cache, re-fetch
    
    min_lat = anchor_lat - bbox_deg
    min_lon = anchor_lon - bbox_deg
    max_lat = anchor_lat + bbox_deg
    max_lon = anchor_lon + bbox_deg
    
    for attempt in range(max_retries + 1):
        try:
            coastlines = _fetch_overpass_tile(min_lat, min_lon, max_lat, max_lon, timeout)
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(coastlines, f)
            if stats is not None:
                stats['coastlines_fetched'] += 1
            return coastlines
        except requests.Timeout:
            if attempt < max_retries:
                log.debug("Coastline fetch timeout for (%.2f, %.2f), retry %d/%d", 
                          anchor_lat, anchor_lon, attempt + 1, max_retries)
                continue
            log.warning("Coastline fetch timeout for (%.2f, %.2f) after %d retries", 
                        anchor_lat, anchor_lon, max_retries)
        except requests.RequestException as exc:
            log.warning("Coastline fetch failed for (%.2f, %.2f): %s", 
                        anchor_lat, anchor_lon, exc)
            break
        except Exception as exc:
            log.warning("Unexpected error fetching coastlines for (%.2f, %.2f): %s", 
                        anchor_lat, anchor_lon, exc)
            break
    
    return []  # Never crash, return empty list on failure

# ---------------------------------------------------------------------------
# Bathymetry / Shallow Water Engine
# ---------------------------------------------------------------------------
class BathymetryGrid:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.depth = data['depth']
        self.lat0 = float(data['lat0'])
        self.lon0 = float(data['lon0'])
        self.dlat = float(data['dlat'])
        self.dlon = float(data['dlon'])

    def depth_at(self, lat, lon):
        i = int(round((lat - self.lat0) / self.dlat))
        j = int(round((lon - self.lon0) / self.dlon))
        if 0 <= i < self.depth.shape[0] and 0 <= j < self.depth.shape[1]:
            d = self.depth[i, j]
            return None if np.isnan(d) else float(d)
        return "OUT_OF_BOUNDS"

    def cells_in_box(self, min_lat, min_lon, max_lat, max_lon):
        i0 = max(0, int((min_lat - self.lat0) / self.dlat))
        i1 = min(self.depth.shape[0], int((max_lat - self.lat0) / self.dlat) + 1)
        j0 = max(0, int((min_lon - self.lon0) / self.dlon))
        j1 = min(self.depth.shape[1], int((max_lon - self.lon0) / self.dlon) + 1)
        cells = []
        for i in range(i0, i1):
            for j in range(j0, j1):
                d = self.depth[i, j]
                if not np.isnan(d):
                    cells.append((self.lat0 + i * self.dlat, self.lon0 + j * self.dlon, float(d)))
        return cells

def add_shallow_waters(scenario, bathymetry, transformer, min_lat, min_lon, max_lat, max_lon,
                       shallow_threshold_m, waters_id_start):
    if bathymetry is None:
        return waters_id_start
    wid = waters_id_start
    half_lat, half_lon = bathymetry.dlat / 2.0, bathymetry.dlon / 2.0
    for lat, lon, depth in bathymetry.cells_in_box(min_lat, min_lon, max_lat, max_lon):
        if depth >= shallow_threshold_m:
            continue
        corners_latlon = [(lat - half_lat, lon - half_lon), (lat - half_lat, lon + half_lon),
                           (lat + half_lat, lon + half_lon), (lat + half_lat, lon - half_lon)]
        pts = np.array([transformer.transform(clon, clat) for clat, clon in corners_latlon])
        scenario.add_objects(Shallow(shape=CRPolygon(pts), waters_id=wid, depth=float(depth)))
        wid += 1
    return wid

def check_grounding(bathymetry, lats, lons, draft_m, stats=None):
    for lat, lon in zip(lats, lons):
        if is_in_exclusion_zone(lat, lon):
            return True
            
        if bathymetry is not None:
            d = bathymetry.depth_at(lat, lon)
            if d == "OUT_OF_BOUNDS":
                if stats is not None:
                    stats['bathymetry_out_of_bounds_points'] += 1
                continue 
            if d is None:
                return True 
            if d < draft_m:
                return True
    return False

# ---------------------------------------------------------------------------
# High-Performance Pure-NumPy Kinematics & Resampling
# ---------------------------------------------------------------------------
def fast_resample_vessel(t_sec, lats, lons, sogs, cogs, hdgs, vessel_type, length, width, draft,
                         dt_seconds=DT_SECONDS, max_gap_bins=MAX_GAP_BINS,
                         min_len=MIN_TRAJECTORY_LEN, stats=None):

    order = np.argsort(t_sec)
    t_sec, lats, lons, sogs, cogs, hdgs = (t_sec[order], lats[order], lons[order],
                                            sogs[order], cogs[order], hdgs[order])

    unique_mask = np.empty(len(t_sec), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = t_sec[1:] != t_sec[:-1]
    t_sec, lats, lons, sogs, cogs, hdgs = (t_sec[unique_mask], lats[unique_mask], lons[unique_mask],
                                            sogs[unique_mask], cogs[unique_mask], hdgs[unique_mask])

    missing_hdg = np.isnan(hdgs) | (hdgs == 511)
    if stats is not None and np.any(missing_hdg):
        stats['heading_imputed_from_cog'] += int(np.sum(missing_hdg))
    hdgs = np.where(missing_hdg, cogs, hdgs)

    valid = (~np.isnan(lats)) & (~np.isnan(lons)) & (~np.isnan(sogs)) & (~np.isnan(cogs)) & (~np.isnan(hdgs))
    t_sec, lats, lons, sogs, cogs, hdgs = t_sec[valid], lats[valid], lons[valid], sogs[valid], cogs[valid], hdgs[valid]

    if len(t_sec) < 2:
        if stats is not None:
            stats['skipped_too_few_raw_rows'] += 1
        return None

    gap_limit = dt_seconds * (max_gap_bins + 1.5)
    gaps = np.diff(t_sec)
    split_indices = np.where(gaps > gap_limit)[0] + 1
    segments = np.split(np.arange(len(t_sec)), split_indices)
    best_seg = max(segments, key=len)
    
    if len(best_seg) < 2:
        if stats is not None:
            stats['skipped_too_few_raw_rows'] += 1
        return None

    t_sec, lats, lons, sogs, cogs, hdgs = (t_sec[best_seg], lats[best_seg], lons[best_seg],
                                            sogs[best_seg], cogs[best_seg], hdgs[best_seg])

    t_start = np.ceil(t_sec[0] / dt_seconds) * dt_seconds
    t_end = np.floor(t_sec[-1] / dt_seconds) * dt_seconds
    if t_end < t_start:
        if stats is not None:
            stats['skipped_too_few_after_resample'] += 1
        return None
        
    t_grid = np.arange(t_start, t_end + (dt_seconds / 2.0), dt_seconds)

    gap_cap_seconds = dt_seconds * max_gap_bins
    idx_right = np.clip(np.searchsorted(t_sec, t_grid), 1, len(t_sec) - 1)
    idx_left = idx_right - 1
    interval_widths = t_sec[idx_right] - t_sec[idx_left]
    grid_ok = interval_widths <= gap_cap_seconds

    if not np.all(grid_ok):
        padded = np.concatenate(([0], grid_ok.astype(int), [0]))
        change = np.diff(padded)
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0]
        if len(starts) == 0:
            if stats is not None:
                stats['skipped_too_few_after_resample'] += 1
            return None
        best = np.argmax(ends - starts)
        t_grid = t_grid[starts[best]:ends[best]]

    if len(t_grid) < min_len:
        if stats is not None:
            stats['skipped_too_few_after_resample'] += 1
        return None

    hdg_unwrapped = np.unwrap(np.deg2rad(hdgs))
    cog_unwrapped = np.unwrap(np.deg2rad(cogs))

    interp_lat = np.interp(t_grid, t_sec, lats)
    interp_lon = np.interp(t_grid, t_sec, lons)
    interp_hdg_unwrapped = np.interp(t_grid, t_sec, hdg_unwrapped)
    interp_cog_unwrapped = np.interp(t_grid, t_sec, cog_unwrapped)

    if stats is not None:
        is_exact_raw = np.isin(t_grid, t_sec)
        stats['resample_real_bins'] += int(np.sum(is_exact_raw))
        stats['resample_fabricated_bins'] += int(np.sum(~is_exact_raw))

    interp_hdg = np.rad2deg(interp_hdg_unwrapped) % 360.0
    interp_cog = np.rad2deg(interp_cog_unwrapped) % 360.0

    # Derive velocity strictly from xy deltas
    dx = np.diff(interp_lon) * 111000.0 * np.cos(np.deg2rad(np.mean(interp_lat)))
    dy = np.diff(interp_lat) * 111000.0
    derived_sog_ms = np.hypot(dx, dy) / dt_seconds
    
    if len(derived_sog_ms) > 0:
        sog_ms = np.append(derived_sog_ms, derived_sog_ms[-1])
    else:
        sog_ms = np.zeros(len(t_grid))

    yaw_rate_r = np.empty_like(interp_hdg_unwrapped)
    yaw_rate_r[1:] = np.diff(interp_hdg_unwrapped) / dt_seconds
    yaw_rate_r[0] = yaw_rate_r[1] if len(yaw_rate_r) > 1 else 0.0

    drift_angle_rad = np.deg2rad(interp_cog - interp_hdg)
    surge_u = sog_ms * np.cos(drift_angle_rad)
    sway_v = sog_ms * np.sin(drift_angle_rad)

    return {
        't_sec': t_grid, 'latitude': interp_lat, 'longitude': interp_lon,
        'sog_ms': sog_ms, 'heading': interp_hdg, 'surge_u': surge_u, 'sway_v': sway_v,
        'yaw_rate_r': yaw_rate_r, 'vessel_type': vessel_type, 'length': length,
        'width': width, 'draft': draft,
    }

def filter_cpa_collisions(valid_groups, safe_cpa_meters=SAFE_CPA_METERS, stats=None, anchor_mmsi=None):
    mmsis_to_drop = set()
    mmsi_list = list(valid_groups.keys())
    for i in range(len(mmsi_list)):
        for j in range(i + 1, len(mmsi_list)):
            m1, m2 = mmsi_list[i], mmsi_list[j]
            if m1 in mmsis_to_drop or m2 in mmsis_to_drop:
                continue
                
            d1, d2 = valid_groups[m1], valid_groups[m2]
            
            common_times, idx1, idx2 = np.intersect1d(d1['t_sec'], d2['t_sec'], return_indices=True)
            if len(common_times) < 2:
                continue
                
            lat1, lon1 = d1['latitude'][idx1], d1['longitude'][idx1]
            lat2, lon2 = d2['latitude'][idx2], d2['longitude'][idx2]
            
            dx = (lon1 - lon2) * 111000.0 * np.cos(np.deg2rad(np.mean(lat1)))
            dy = (lat1 - lat2) * 111000.0
            actual_distances = np.hypot(dx, dy)
            
            hull_buffer = (d1['length'] + d2['length']) / 2.0
            actual_threshold = max(safe_cpa_meters, hull_buffer)
            
            if np.any(actual_distances < actual_threshold):
                # Never drop anchor
                if anchor_mmsi is not None and (m1 == anchor_mmsi or m2 == anchor_mmsi):
                    # Drop the other ship instead
                    dropped = m2 if m1 == anchor_mmsi else m1
                else:
                    dropped = m2 if d1['length'] > d2['length'] else m1
                mmsis_to_drop.add(dropped)
                if stats is not None:
                    stats['cpa_collisions_removed'] += 1
                    
    for m in mmsis_to_drop:
        del valid_groups[m]
    return valid_groups

def _log_summary(stats):
    log.info("----- pipeline summary -----")
    for key in ['rows_excluded_zone', 'windows_seen', 'anchors_considered', 'skipped_anchor_reused',
                'skipped_location_overused', 'skipped_anchor_too_few_rows', 'skipped_too_few_raw_rows', 
                'skipped_temporal_desync', # <-- NEW: Tracks ships dropped due to misaligned timestamps
                'boxes_downsampled', 'skipped_too_few_after_resample', 'cpa_collisions_removed',
                'skipped_too_few_after_collision_filter', 'skipped_anchor_dropped_by_cpa',
                'skipped_due_to_grounding', 'skipped_anchor_grounded',
                'skipped_boring_straight_lines', 'skipped_insufficient_large_movers',
                'skipped_redundant_fleet', 'grounding_flags', 'bathymetry_out_of_bounds_points',
                'scenarios_written', 'scenarios_written_moving_mode', 'scenarios_written_density_mode',
                'coastlines_fetched', 'coastlines_cached',
                # --- NEW: Large ship / ego stats ---
                'windows_skipped_insufficient_large_movers',
                'scenarios_ego_large', 'scenarios_ego_small', 'scenarios_ego_anchored']:
        log.info("%s: %d", key, stats.get(key, 0))
    real = stats.get('resample_real_bins', 0)
    fab = stats.get('resample_fabricated_bins', 0)
    if real + fab:
        log.info("resampled bins that were interpolated rather than real: %.1f%% (%d of %d)",
                  100 * fab / (real + fab), fab, real + fab)
    # Large ship distribution summary
    total_scenes = stats.get('scenarios_written', 0)
    if total_scenes > 0:
        log.info("Ego distribution: large=%.1f%%, small=%.1f%%, anchored=%.1f%%",
                  100*stats.get('scenarios_ego_large',0)/total_scenes,
                  100*stats.get('scenarios_ego_small',0)/total_scenes,
                  100*stats.get('scenarios_ego_anchored',0)/total_scenes)
        log.info("Avg large movers per scene: %.1f",
                  stats.get('large_movers_per_scene_sum',0)/total_scenes)

# ---------------------------------------------------------------------------
# Main Pipeline Engine
# ---------------------------------------------------------------------------
def generate_dataset_from_ais(zst_filepath, output_dir="/run/media/akshat/Akshat_USB/generated_scenarios_LA",
                               sampling_strategy="density_first", max_windows=None, manifest_path="manifest.jsonl",
                               seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    stats = Counter()
    manifest_entries = []
    location_usage = Counter()

    columns_to_load = ['mmsi', 'base_date_time', 'latitude', 'longitude', 'sog', 'cog',
                        'heading', 'vessel_type', 'length', 'width', 'draft']
    df = pd.read_csv(zst_filepath, compression='zstd', usecols=columns_to_load)
    df['base_date_time'] = pd.to_datetime(df['base_date_time'])
    df = df.dropna(subset=['latitude', 'longitude'])

    # Drop sentinel errors
    df = df[df['sog'] < 102.0]
    df = apply_exclusion_zones(df)

    length_missing = df['length'].isna().mean()
    width_missing = df['width'].isna().mean()
    draft_missing = df['draft'].isna().mean()
    log.info("Data ingested. Length missing for %.1f%%, width for %.1f%%, draft for %.1f%% (imputing to %.0fx%.0fm, %.0fm draft).",
             100 * length_missing, 100 * width_missing, 100 * draft_missing, DEFAULT_LENGTH_M, DEFAULT_WIDTH_M, DEFAULT_DRAFT_M)

    df['length'] = df['length'].fillna(DEFAULT_LENGTH_M)
    df['width'] = df['width'].fillna(DEFAULT_WIDTH_M)
    df['draft'] = df['draft'].fillna(DEFAULT_DRAFT_M)

    if df.empty:
        log.warning("No rows left after cleaning.")
        return stats

    bathymetry = BathymetryGrid(BATHYMETRY_NPZ_PATH) if BATHYMETRY_NPZ_PATH else None
    
    # Pre-load offline coastlines if enabled (loads once, caches in memory)
    if USE_OFFLINE_COASTLINES:
        all_coastlines = load_offline_coastlines()
        log.info("Offline coastline mode enabled: %d segments loaded", len(all_coastlines))
    else:
        # Coastline cache is now per-scenario (lazy loading), no upfront fetch needed
        all_coastlines = []  # Kept for compatibility, not used when COASTLINE_PER_SCENARIO=True

    df = df.sort_values('base_date_time').reset_index(drop=True)
    timestamps_sec = (df['base_date_time'].values.astype('datetime64[s]').astype(np.int64))

    start_sec = timestamps_sec[0]
    end_sec = timestamps_sec[-1]
    total_minutes = int((end_sec - start_sec) // 60)
    if max_windows is not None:
        total_minutes = min(total_minutes, max_windows * WINDOW_STEP_MINUTES)

    scenario_counter = 0
    exported_scenario_fleets = []
    vessel_to_exported_fleets = defaultdict(list)
    last_start_min = max(total_minutes - WINDOW_MINUTES, 0)

    for min_offset in range(0, last_start_min + 1, WINDOW_STEP_MINUTES):
        t0_sec = start_sec + (min_offset * 60)
        t1_sec = t0_sec + (WINDOW_MINUTES * 60)

        idx0 = np.searchsorted(timestamps_sec, t0_sec, side='left')
        idx1 = np.searchsorted(timestamps_sec, t1_sec, side='left')
        if idx1 <= idx0:
            continue

        w_df = df.iloc[idx0:idx1]
        stats['windows_seen'] += 1

        w_mmsi = w_df['mmsi'].values
        w_lat = w_df['latitude'].values
        w_lon = w_df['longitude'].values
        w_sog = w_df['sog'].values
        w_cog = w_df['cog'].values
        w_hdg = w_df['heading'].values
        w_vtype = w_df['vessel_type'].values
        w_len = w_df['length'].values
        w_wid = w_df['width'].values
        w_draft = w_df['draft'].values
        w_tsec = timestamps_sec[idx0:idx1]

        unique_mmsis, inv_indices = np.unique(w_mmsi, return_inverse=True)
        num_unique = len(unique_mmsis)
        if num_unique < 2:
            continue

        mmsi_mean_lat = np.zeros(num_unique, dtype=np.float64)
        mmsi_mean_lon = np.zeros(num_unique, dtype=np.float64)
        mmsi_counts = np.bincount(inv_indices, minlength=num_unique)

        np.add.at(mmsi_mean_lat, inv_indices, w_lat)
        np.add.at(mmsi_mean_lon, inv_indices, w_lon)
        mmsi_mean_lat /= mmsi_counts
        mmsi_mean_lon /= mmsi_counts

        coords = np.column_stack((mmsi_mean_lat, mmsi_mean_lon))
        tree = cKDTree(coords)


        # --- MIXED SAMPLING: Decide path for this window ---
        use_strict = False
        if SAMPLING_MODE == "strict":
            use_strict = True
        elif SAMPLING_MODE == "relaxed":
            use_strict = False
        elif SAMPLING_MODE == "mixed":
            use_strict = (np.random.rand() < STRICT_SAMPLING_RATIO)

        # --- COMMON: Compute stats for both paths ---
        mmsi_mean_sog = np.zeros(num_unique, dtype=np.float64)
        mmsi_max_len = np.zeros(num_unique, dtype=np.float64)
        np.add.at(mmsi_mean_sog, inv_indices, w_sog)
        np.maximum.at(mmsi_max_len, inv_indices, w_len)
        mmsi_mean_sog /= mmsi_counts
        mean_speed_knots = mmsi_mean_sog / 0.514444

        mmsi_vtype = np.empty(num_unique, dtype=object)
        mmsi_vtype[inv_indices] = w_vtype
        is_large = mmsi_max_len > LARGE_SHIP_LENGTH_M
        is_cargo_type = np.array([is_cargo_or_tanker(code) for code in mmsi_vtype], dtype=bool)
        is_fast_enough = mean_speed_knots > CARGO_SPEED_KNOTS
        has_enough_data = mmsi_counts >= 2
        large_mover_mask = is_large & is_cargo_type & is_fast_enough & has_enough_data
        n_large_movers_in_window = int(np.sum(large_mover_mask))

        grid_neighbors = tree.query_ball_point(coords, r=0.1, p=np.inf)
        densities = np.array([len(nbrs) for nbrs in grid_neighbors])

        if use_strict:
            # ===== STRICT PATH: Large-mover-first =====
            # Skip window if it cannot yield enough large movers
            if n_large_movers_in_window < TARGET_LARGE_MOVERS_PER_SCENE:
                stats['windows_skipped_insufficient_large_movers'] += 1
                continue

            # Score large movers as ego candidates by neighborhood richness
            ego_candidate_mask = large_mover_mask
            if not np.any(ego_candidate_mask):
                stats['skipped_anchor_too_few_rows'] += 1
                continue

            ego_scores = np.zeros(num_unique, dtype=np.float64)
            for i in np.where(ego_candidate_mask)[0]:
                nbrs = tree.query_ball_point(coords[i], r=LOCAL_BOX_DEG, p=np.inf)
                n_large_nbrs = int(np.sum(large_mover_mask[nbrs]))
                ego_scores[i] = n_large_nbrs * (1 + densities[i]) * (1 + mmsi_max_len[i] / 100.0)

            anchor_order = np.argsort(-ego_scores[ego_candidate_mask])
            ranked_anchor_indices = np.where(ego_candidate_mask)[0][anchor_order]
            focus_on_moving_cargo = True  # for compatibility with downstream logic
        else:
            # ===== RELAXED PATH: Density-first (original logic) =====
            # 80% chance to hunt for big moving ships, 20% chance to fallback to raw density
            focus_on_moving_cargo = np.random.rand() < 0.80

            if focus_on_moving_cargo:
                # 80% Mode: Exploit (Prioritize massive, moving cargo ships)
                moving_mask = mean_speed_knots > MIN_DYNAMIC_SPEED_KNOTS
                valid_anchor_mask = (densities > 1) & (mmsi_counts >= 2) & moving_mask
                
                if not np.any(valid_anchor_mask):
                    continue
                    
                anchor_scores = densities[valid_anchor_mask] * mmsi_max_len[valid_anchor_mask]
                anchor_order = np.argsort(-anchor_scores)
                ranked_anchor_indices = np.where(valid_anchor_mask)[0][anchor_order]
                
            else:
                # 20% Mode: Explore (Fallback to pure density for anchored fleets & small craft)
                valid_anchor_mask = (densities > 1) & (mmsi_counts >= 2)
                if not np.any(valid_anchor_mask):
                    continue
                    
                anchor_scores = densities[valid_anchor_mask]
                anchor_order = np.argsort(-anchor_scores)
                ranked_anchor_indices = np.where(valid_anchor_mask)[0][anchor_order]
            # -------------------------------------------------------------

        vessel_indices_map = defaultdict(list)
        for i_row, m_idx in enumerate(inv_indices):
            vessel_indices_map[m_idx].append(i_row)

        window_resampled_cache = {}

        for anchor_idx in ranked_anchor_indices:
            anchor_mmsi = unique_mmsis[anchor_idx]
            stats['anchors_considered'] += 1

            if any(anchor_mmsi in exported_scenario_fleets[fid] for fid in vessel_to_exported_fleets.get(anchor_mmsi, [])):
                stats['skipped_anchor_reused'] += 1
                continue

            anchor_mean_coord = coords[anchor_idx]
            neighbor_indices = tree.query_ball_point(anchor_mean_coord, r=LOCAL_BOX_DEG, p=np.inf)
            if len(neighbor_indices) < 2:
                stats['skipped_too_few_after_resample'] += 1
                continue

            valid_groups = {}
            for n_idx in neighbor_indices:
                if mmsi_counts[n_idx] < 2:
                    continue
                mmsi_val = unique_mmsis[n_idx]
                if mmsi_val not in window_resampled_cache:
                    rows = vessel_indices_map[n_idx]
                    res = fast_resample_vessel(
                        t_sec=w_tsec[rows], lats=w_lat[rows], lons=w_lon[rows],
                        sogs=w_sog[rows], cogs=w_cog[rows], hdgs=w_hdg[rows],
                        vessel_type=w_vtype[rows[0]], length=w_len[rows[0]],
                        width=w_wid[rows[0]], draft=w_draft[rows[0]],
                        dt_seconds=DT_SECONDS, max_gap_bins=MAX_GAP_BINS,
                        min_len=MIN_TRAJECTORY_LEN, stats=stats
                    )
                    window_resampled_cache[mmsi_val] = res
                if window_resampled_cache[mmsi_val] is not None:
                    valid_groups[mmsi_val] = window_resampled_cache[mmsi_val]

            if anchor_mmsi not in valid_groups:
                stats['skipped_anchor_too_few_rows'] += 1
                continue
            if len(valid_groups) < 2:
                stats['skipped_too_few_after_resample'] += 1
                continue

            # --- FIX 1: THE UNIVERSAL SCENE CLOCK ---

            # --- DYNAMIC TIME STEP SELECTION ---
            # --- FIX 2: ADAPTIVE TIME STEP SELECTION ---
            anchor_res = valid_groups[anchor_mmsi]
            total_available_frames = len(anchor_res['t_sec'])
            
            if total_available_frames < TARGET_FRAMES:
                stats['skipped_too_few_after_resample'] += 1
                continue
                
            best_start_idx = 0
            highest_speed = 0.0
            for i in range(total_available_frames - TARGET_FRAMES + 1):
                window_speed = np.mean(anchor_res['sog_ms'][i : i + TARGET_FRAMES])
                if window_speed > highest_speed:
                    highest_speed = window_speed
                    best_start_idx = i
                        
            scene_t_sec = anchor_res['t_sec'][best_start_idx : best_start_idx + TARGET_FRAMES]
            # -------------------------------------------

            # --- FIX 2: TEMPORAL SYNCHRONIZATION ---
            # Throw out any ship that doesn't exist during those exact 20 timestamps
            mmsis_to_drop_time = set()
            for m, res in valid_groups.items():
                if not np.all(np.isin(scene_t_sec, res['t_sec'])):
                    mmsis_to_drop_time.add(m)
            
            for m in mmsis_to_drop_time:
                del valid_groups[m]
                
            if len(valid_groups) < 2 or anchor_mmsi not in valid_groups:
                stats['skipped_temporal_desync'] += 1
                continue

            # Lock the mapping origin strictly to the anchor's synchronized physical starting position
            s_idx_anchor = np.searchsorted(valid_groups[anchor_mmsi]['t_sec'], scene_t_sec[0])
            origin_lat = valid_groups[anchor_mmsi]['latitude'][s_idx_anchor]
            origin_lon = valid_groups[anchor_mmsi]['longitude'][s_idx_anchor]

            grid_key = (round(origin_lat, 2), round(origin_lon, 2))
            if location_usage[grid_key] >= MAX_SCENARIOS_PER_GRID_CELL:
                stats['skipped_location_overused'] += 1
                continue

            if len(valid_groups) > MAX_SHIPS_PER_SCENE:
                stats['boxes_downsampled'] += 1
                # Priority-based keep: ego + large movers + close others
                ship_priorities = []
                for m in valid_groups:
                    res = valid_groups[m]
                    is_ego = (m == anchor_mmsi)
                    is_large = res['length'] > LARGE_SHIP_LENGTH_M
                    is_cargo_type = is_cargo_or_tanker(res['vessel_type'])
                    mean_sog_knots = np.mean(res['sog_ms']) / 0.514444
                    is_fast_cargo = is_large and is_cargo_type and mean_sog_knots > CARGO_SPEED_KNOTS
                    
                    v_lat = np.mean(res['latitude'])
                    v_lon = np.mean(res['longitude'])
                    dist = np.hypot(v_lat - origin_lat, v_lon - origin_lon)
                    
                    # Priority: ego (0) > large fast cargo (1) > large (2) > others by distance (3+)
                    if is_ego:
                        priority = 0
                    elif is_fast_cargo:
                        priority = 1
                    elif is_large:
                        priority = 2
                    else:
                        priority = 3 + dist * 1000  # Distance tiebreaker
                    
                    ship_priorities.append((priority, dist, m))
                
                ship_priorities.sort(key=lambda x: (x[0], x[1]))
                keep = set([x[2] for x in ship_priorities[:MAX_SHIPS_PER_SCENE]])
                valid_groups = {m: valid_groups[m] for m in keep}

            valid_groups = filter_cpa_collisions(valid_groups, stats=stats, anchor_mmsi=anchor_mmsi)
            if len(valid_groups) < 2:
                stats['skipped_too_few_after_collision_filter'] += 1
                continue
            if anchor_mmsi not in valid_groups:
                stats['skipped_anchor_dropped_by_cpa'] += 1
                continue

            # --- MANEUVER FILTER ---
       # --- FIX 3: ADAPTIVE MANEUVER & DISPLACEMENT FILTER (LARGE-SHIP AWARE) ---
            dynamic_ship_count = 0
            meaningful_movement_count = 0
            large_mover_count = 0
            
            for m, res in valid_groups.items():
                s_idx = np.searchsorted(res['t_sec'], scene_t_sec[0])
                sliced_r = res['yaw_rate_r'][s_idx : s_idx + TARGET_FRAMES]
                sliced_sog = res['sog_ms'][s_idx : s_idx + TARGET_FRAMES]
                
                # 1. Calculate physical straight-line displacement (A to B)
                start_lat = res['latitude'][s_idx]
                start_lon = res['longitude'][s_idx]
                end_lat = res['latitude'][s_idx + TARGET_FRAMES - 1]
                end_lon = res['longitude'][s_idx + TARGET_FRAMES - 1]
                
                dx = (end_lon - start_lon) * 111000.0 * np.cos(np.deg2rad(start_lat))
                dy = (end_lat - start_lat) * 111000.0
                displacement_m = np.hypot(dx, dy)
                
                if displacement_m > MIN_DISPLACEMENT_METERS:
                    meaningful_movement_count += 1
                
                # 2. Calculate dynamic kinematics
                mean_speed_knots = np.mean(sliced_sog) / 0.514444
                total_turn_deg = np.sum(np.abs(sliced_r)) * DT_SECONDS * (180.0 / np.pi)
                
                # Check if this is a large cargo ship in transit
                is_large = res['length'] > LARGE_SHIP_LENGTH_M
                is_cargo_type = is_cargo_or_tanker(res['vessel_type'])
                is_large_cargo_transit = is_large and is_cargo_type and mean_speed_knots > CARGO_SPEED_KNOTS
                
                # Large cargo in transit counts as dynamic even without significant turn
                if is_large_cargo_transit:
                    dynamic_ship_count += 1
                    large_mover_count += 1
                elif mean_speed_knots > MIN_DYNAMIC_SPEED_KNOTS and total_turn_deg > MIN_TURN_DEG_PER_SCENE:
                    dynamic_ship_count += 1
                    
            # RULE 1 (Universal Ban): 
            # If NO ship travels at least 50 meters, the entire scene is just jitter/anchored. Delete it.
            if meaningful_movement_count < 1:
                stats['skipped_boring_straight_lines'] += 1
                continue
                
            # RULE 2: Require at least 2 dynamic ships (strict mode) or 1 (relaxed mode)
            min_dynamic = 2 if use_strict else 1
            if dynamic_ship_count < min_dynamic:
                stats['skipped_boring_straight_lines'] += 1
                continue
            
            # RULE 3: In strict mode, require large movers; relaxed mode skips this
            if use_strict and large_mover_count < TARGET_LARGE_MOVERS_PER_SCENE:
                stats['skipped_insufficient_large_movers'] += 1
                continue
            # ------------------------------------------------------

            mmsis_to_drop_grounding = set()
            for m, res in valid_groups.items():
                if check_grounding(bathymetry, res['latitude'], res['longitude'], float(res['draft']), stats=stats):
                    mmsis_to_drop_grounding.add(m)
                    stats['grounding_flags'] += 1
            for m in mmsis_to_drop_grounding:
                del valid_groups[m]

            if len(valid_groups) < 2 or anchor_mmsi not in valid_groups:
                stats['skipped_due_to_grounding'] += 1
                continue

            current_fleet = set(valid_groups.keys())
            candidate_fleet_ids = set()
            for m in current_fleet:
                candidate_fleet_ids.update(vessel_to_exported_fleets.get(m, []))
            is_redundant = False
            for fid in candidate_fleet_ids:
                past_fleet = exported_scenario_fleets[fid]
                overlap = len(current_fleet & past_fleet)
                if overlap > 0 and (overlap / min(len(current_fleet), len(past_fleet))) > FLEET_OVERLAP_THRESHOLD:
                    is_redundant = True
                    break
            if is_redundant:
                stats['skipped_redundant_fleet'] += 1
                continue

            # --- NEW: Per-scenario coastline fetching (fast, cached, deterministic) ---
            if COASTLINE_PER_SCENARIO:
                coastlines = fetch_coastlines_for_scenario(origin_lat, origin_lon, stats=stats)
            else:
                coastlines = coastlines_near(
                    all_coastlines, origin_lat - LOCAL_BOX_DEG, origin_lon - LOCAL_BOX_DEG,
                    origin_lat + LOCAL_BOX_DEG, origin_lon + LOCAL_BOX_DEG
                )

            transformer = Transformer.from_crs(
                "epsg:4326", f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +units=m", always_xy=True
            )
            scenario = Scenario(dt=DT_SECONDS, scenario_id=f"ZAM_Batch-{scenario_counter:04d}_1_T-1")

            for coastline in coastlines:
                if len(coastline) < 2:
                    continue
                projected = [transformer.transform(lon, lat) for lon, lat in coastline]
                scenario.add_objects(StaticObstacle(
                    obstacle_id=scenario.generate_object_id(),
                    obstacle_type=ObstacleType.LAND,
                    obstacle_shape=CRPolygon(np.array(LineString(projected).buffer(20.0).exterior.coords)),
                    initial_state=InitialState(position=np.array([0, 0]), orientation=0.0, time_step=0)
                ))

            if bathymetry is not None:
                max_draft = max([res['draft'] for res in valid_groups.values()])
                add_shallow_waters(
                    scenario, bathymetry, transformer,
                    origin_lat - LOCAL_BOX_DEG, origin_lon - LOCAL_BOX_DEG,
                    origin_lat + LOCAL_BOX_DEG, origin_lon + LOCAL_BOX_DEG,
                    shallow_threshold_m=max_draft + SHALLOW_SAFETY_MARGIN_M,
                    waters_id_start=10000,
                )

            n_obstacles_added = 0
            for mmsi_val, res in valid_groups.items():
                ship_length = float(res['length'])
                ship_width = float(res['width'])
                ship_draft = float(res['draft'])
                obstacle_type = vessel_type_to_obstacle_type(res['vessel_type'], np.mean(res['sog_ms']))

                s_idx = np.searchsorted(res['t_sec'], scene_t_sec[0])
                
                state_list = []
                trajectory_is_valid = True
                
                for f_idx in range(TARGET_FRAMES):
                    local_idx = s_idx + f_idx
                    
                    # --- FIX 3: ZERO-INDEXING TENSORS ---
                    # Forces time_step to be 0, 1, 2, 3... perfectly synced for all ships
                    exact_time_step = f_idx  
                    
                    x, y = transformer.transform(res['longitude'][local_idx], res['latitude'][local_idx])
                    
                    if abs(x) > MAX_SCENE_RADIUS_M or abs(y) > MAX_SCENE_RADIUS_M:
                        trajectory_is_valid = False
                        break
                        
                    heading_rad = np.deg2rad((90.0 - res['heading'][local_idx]) % 360.0)
                    state_list.append(TFState(
                        position=np.array([x, y]), orientation=heading_rad,
                        velocity=res['surge_u'][local_idx], velocity_y=res['sway_v'][local_idx],
                        yaw_rate=res['yaw_rate_r'][local_idx], time_step=exact_time_step
                    ))

                if not trajectory_is_valid or len(state_list) != TARGET_FRAMES:
                    continue
                    
                initial_time_step = state_list[0].time_step
                initial_state = InitialState(
                    position=state_list[0].position, velocity=state_list[0].velocity,
                    velocity_y=state_list[0].velocity_y, orientation=state_list[0].orientation,
                    yaw_rate=state_list[0].yaw_rate, time_step=initial_time_step
                )
                
                shape = Rectangle(length=ship_length, width=ship_width)
                scenario.add_objects(DynamicObstacle(
                    obstacle_id=scenario.generate_object_id(), obstacle_type=obstacle_type,
                    obstacle_shape=shape, initial_state=initial_state,
                    prediction=TrajectoryPrediction(Trajectory(initial_time_step + 1, state_list[1:]), shape),
                    depth=ship_draft,
                ))
                n_obstacles_added += 1

            if n_obstacles_added > 1:
                fleet_id = len(exported_scenario_fleets)
                exported_scenario_fleets.append(current_fleet)
                for m in current_fleet:
                    vessel_to_exported_fleets[m].append(fleet_id)

                location_usage[grid_key] += 1

                output_file = os.path.join(output_dir, f"scenario_{scenario_counter:04d}.xml")
                CommonOceanFileWriter(
                    scenario=scenario, planning_problem_set=PlanningProblemSet(),
                    author="Batch ML Pipeline", affiliation="Data Engineering", source="AIS", tags={Tag.OPENSEA}
                ).write_to_file(output_file, OverwriteExistingFile.ALWAYS)

                manifest_entries.append(dict(
                    scenario_id=f"scenario_{scenario_counter:04d}",
                    t_start=str(pd.to_datetime(scene_t_sec[0], unit='s')),
                    t_end=str(pd.to_datetime(scene_t_sec[-1], unit='s')),
                    origin_lat=origin_lat, origin_lon=origin_lon,
                    mmsis=sorted(int(m) for m in current_fleet),
                    n_vessels=n_obstacles_added,
                ))

                # --- NEW: Track ego distribution for diffusion training ---
                anchor_res = valid_groups[anchor_mmsi]
                anchor_len = anchor_res['length']
                anchor_mean_sog_knots = np.mean(anchor_res['sog_ms']) / 0.514444
                
                if (anchor_len > LARGE_SHIP_LENGTH_M and
                        is_cargo_or_tanker(anchor_res['vessel_type']) and
                        anchor_mean_sog_knots > CARGO_SPEED_KNOTS):
                    stats['scenarios_ego_large'] += 1
                elif anchor_mean_sog_knots > MIN_DYNAMIC_SPEED_KNOTS:
                    stats['scenarios_ego_small'] += 1
                else:
                    stats['scenarios_ego_anchored'] += 1
                
                # Track large movers per scene for averaging
                stats['large_movers_per_scene_sum'] = stats.get('large_movers_per_scene_sum', 0) + large_mover_count
                
                stats['scenarios_written'] += 1
                # Note: focus_on_moving_cargo no longer used; all scenarios use ego-centric selection
                stats['scenarios_written_moving_mode'] += 1  # Keep for compatibility
                
                log.info("[%d] wrote %s with %d vessels (Ego: len=%.0f, sog=%.1fkt, LargeMovers=%d)", 
                         scenario_counter, output_file, n_obstacles_added, anchor_len, anchor_mean_sog_knots, large_mover_count)
                scenario_counter += 1
                # ------------------------------------------------------
    if manifest_entries:
        with open(manifest_path, "w") as f:
            for e in manifest_entries:
                f.write(json.dumps(e) + "\n")
        log.info("Wrote %d entries to %s", len(manifest_entries), manifest_path)

    _log_summary(stats)
    return stats

if __name__ == "__main__":
    generate_dataset_from_ais("/run/media/akshat/Akshat_USB/AISFiles/LA_ais.csv.zst", sampling_strategy="density_first")