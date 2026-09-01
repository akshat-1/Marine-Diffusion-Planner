import os
import sys
import json
import logging
import glob
import gc
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import LineString, box
from shapely.strtree import STRtree
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

# Force console logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
log = logging.getLogger("ais_pipeline")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MIN_DISPLACEMENT_METERS = 50.0   
DT_SECONDS = 10.0
WINDOW_MINUTES = 20
WINDOW_STEP_MINUTES = 5
MAX_GAP_BINS = 45                 
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

MIN_TURN_DEG_PER_SCENE = 10.0     
MIN_DYNAMIC_SPEED_KNOTS = 2.0     

LARGE_SHIP_LENGTH_M = 100.0           
CARGO_SPEED_KNOTS = 3.0               
TARGET_LARGE_MOVERS_PER_SCENE = 1     
EGO_LARGE_TARGET_RATIO = 0.6          
SAMPLING_MODE = "mixed"               
STRICT_SAMPLING_RATIO = 0.80          

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
        log.info("Excluding %d rows (%.2f%%) inside EXCLUSION_ZONES.", dropped, (dropped/len(df))*100)
    return df[~mask].reset_index(drop=True)

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
    if stats is not None and np.any(missing_hdg): stats['heading_imputed_from_cog'] += int(np.sum(missing_hdg))
    hdgs = np.where(missing_hdg, cogs, hdgs)

    valid = (~np.isnan(lats)) & (~np.isnan(lons)) & (~np.isnan(sogs)) & (~np.isnan(cogs)) & (~np.isnan(hdgs))
    t_sec, lats, lons, sogs, cogs, hdgs = t_sec[valid], lats[valid], lons[valid], sogs[valid], cogs[valid], hdgs[valid]

    if len(t_sec) < 2: return None

    gap_limit = dt_seconds * (max_gap_bins + 1.5)
    gaps = np.diff(t_sec)
    split_indices = np.where(gaps > gap_limit)[0] + 1
    segments = np.split(np.arange(len(t_sec)), split_indices)
    best_seg = max(segments, key=len)
    
    if len(best_seg) < 2: return None

    t_sec, lats, lons, sogs, cogs, hdgs = (t_sec[best_seg], lats[best_seg], lons[best_seg],
                                            sogs[best_seg], cogs[best_seg], hdgs[best_seg])

    t_start = np.ceil(t_sec[0] / dt_seconds) * dt_seconds
    t_end = np.floor(t_sec[-1] / dt_seconds) * dt_seconds
    if t_end < t_start: return None
        
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
        if len(starts) == 0: return None
        best = np.argmax(ends - starts)
        t_grid = t_grid[starts[best]:ends[best]]

    if len(t_grid) < min_len: return None

    interp_lat = np.interp(t_grid, t_sec, lats)
    interp_lon = np.interp(t_grid, t_sec, lons)
    interp_hdg = np.rad2deg(np.interp(t_grid, t_sec, np.unwrap(np.deg2rad(hdgs)))) % 360.0
    interp_cog = np.rad2deg(np.interp(t_grid, t_sec, np.unwrap(np.deg2rad(cogs)))) % 360.0

    dx = np.diff(interp_lon) * 111000.0 * np.cos(np.deg2rad(np.mean(interp_lat)))
    dy = np.diff(interp_lat) * 111000.0
    derived_sog_ms = np.hypot(dx, dy) / dt_seconds
    sog_ms = np.append(derived_sog_ms, derived_sog_ms[-1]) if len(derived_sog_ms) > 0 else np.zeros(len(t_grid))

    yaw_rate_r = np.empty_like(interp_hdg)
    yaw_rate_r[1:] = np.diff(np.unwrap(np.deg2rad(interp_hdg))) / dt_seconds
    yaw_rate_r[0] = yaw_rate_r[1] if len(yaw_rate_r) > 1 else 0.0

    drift_angle_rad = np.deg2rad(interp_cog - interp_hdg)
    return {
        't_sec': t_grid, 'latitude': interp_lat, 'longitude': interp_lon,
        'sog_ms': sog_ms, 'heading': interp_hdg, 'surge_u': sog_ms * np.cos(drift_angle_rad), 
        'sway_v': sog_ms * np.sin(drift_angle_rad), 'yaw_rate_r': yaw_rate_r, 
        'vessel_type': vessel_type, 'length': length, 'width': width, 'draft': draft,
    }

def filter_cpa_collisions(valid_groups, safe_cpa_meters=SAFE_CPA_METERS, stats=None):
    """
    MODIFIED: Treats all ships equally. Removes the smaller ship in a collision.
    No longer provides "Ego Armor" to the anchor.
    """
    mmsis_to_drop = set()
    mmsi_list = list(valid_groups.keys())
    for i in range(len(mmsi_list)):
        for j in range(i + 1, len(mmsi_list)):
            m1, m2 = mmsi_list[i], mmsi_list[j]
            if m1 in mmsis_to_drop or m2 in mmsis_to_drop:
                continue
                
            d1, d2 = valid_groups[m1], valid_groups[m2]
            common_times, idx1, idx2 = np.intersect1d(d1['t_sec'], d2['t_sec'], return_indices=True)
            if len(common_times) < 2: continue
                
            lat1, lon1 = d1['latitude'][idx1], d1['longitude'][idx1]
            lat2, lon2 = d2['latitude'][idx2], d2['longitude'][idx2]
            
            dx = (lon1 - lon2) * 111000.0 * np.cos(np.deg2rad(np.mean(lat1)))
            dy = (lat1 - lat2) * 111000.0
            actual_distances = np.hypot(dx, dy)
            
            hull_buffer = (d1['length'] + d2['length']) / 2.0
            actual_threshold = max(safe_cpa_meters, hull_buffer)
            
            if np.any(actual_distances < actual_threshold):
                # Standard physics: smaller ship is deleted from scenario
                dropped = m2 if d1['length'] > d2['length'] else m1
                mmsis_to_drop.add(dropped)
                if stats is not None: stats['cpa_collisions_removed'] += 1
                    
    for m in mmsis_to_drop: del valid_groups[m]
    return valid_groups

def _log_summary(stats):
    log.info("\n----- Pipeline Summary -----")
    for key in ['windows_seen', 'anchors_considered', 'skipped_anchor_reused',
                'skipped_location_overused', 'skipped_anchor_too_few_rows', 'skipped_too_few_raw_rows', 
                'skipped_temporal_desync', 'boxes_downsampled', 'skipped_too_few_after_resample', 
                'cpa_collisions_removed', 'skipped_too_few_after_collision_filter',
                'coastline_collisions_prevented', 'skipped_due_to_coastline_collision', 'skipped_boring_straight_lines', 
                'skipped_insufficient_large_movers', 'skipped_redundant_fleet',
                'scenarios_written', 'windows_skipped_insufficient_large_movers',
                'scenarios_ego_large', 'scenarios_ego_small', 'scenarios_ego_anchored']:
        log.info("%s: %d", key, stats.get(key, 0))

# ---------------------------------------------------------------------------
# Main Pipeline Engine
# ---------------------------------------------------------------------------
def generate_dataset_from_ais(zst_filepath, shp_filepath, output_dir, 
                               sampling_strategy="density_first", max_windows=None, manifest_path="manifest.jsonl",
                               seed=42, on_scenario_created=None):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    stats = Counter()
    manifest_entries = []
    location_usage = Counter()

    log.info(f"Loading dataset: {os.path.basename(zst_filepath)}...")
    columns_to_load = ['mmsi', 'base_date_time', 'latitude', 'longitude', 'sog', 'cog',
                        'heading', 'vessel_type', 'length', 'width', 'draft']
    df = pd.read_csv(zst_filepath, compression='zstd', usecols=columns_to_load)
    df['base_date_time'] = pd.to_datetime(df['base_date_time'])
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[df['sog'] < 102.0]
    df = apply_exclusion_zones(df)

    df['length'] = df['length'].fillna(DEFAULT_LENGTH_M)
    df['width'] = df['width'].fillna(DEFAULT_WIDTH_M)
    df['draft'] = df['draft'].fillna(DEFAULT_DRAFT_M)

    if df.empty:
        log.warning("No rows left after cleaning.")
        return stats
    
    # -------------------------------------------------------------
    # PRE-PROCESS COASTLINES FOR THIS SPECIFIC DATASET
    # -------------------------------------------------------------
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    
    log.info("Loading global coastlines from shapefile and clipping to dataset bounds...")
    world_coastlines = gpd.read_file(shp_filepath)
    bbox = box(min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1)
    local_lines = world_coastlines.cx[min_lon - 0.1 : max_lon + 0.1, min_lat - 0.1 : max_lat + 0.1].clip(bbox)
    del world_coastlines
    gc.collect()

    global_lat, global_lon = df['latitude'].mean(), df['longitude'].mean()
    global_transformer = Transformer.from_crs("epsg:4326", f"+proj=aeqd +lat_0={global_lat} +lon_0={global_lon} +units=m", always_xy=True)
    
    global_coastline_polygons = []
    
    log.info("Pre-computing high-speed C-optimized coastline buffers...")
    for geom in local_lines.geometry:
        lines_to_process = []
        if geom.geom_type == 'LineString':
            lines_to_process.append(geom)
        elif geom.geom_type == 'MultiLineString':
            lines_to_process.extend(geom.geoms)
            
        for line in lines_to_process:
            if len(line.coords) < 2: continue
            line_arr = np.array(line.coords)
            xs, ys = global_transformer.transform(line_arr[:, 0], line_arr[:, 1])
            projected_line = LineString(np.column_stack((xs, ys)))
            
            buffered_poly = projected_line.buffer(20.0, resolution=2).simplify(5.0, preserve_topology=True)
            
            if buffered_poly.geom_type == 'Polygon':
                global_coastline_polygons.append(buffered_poly)
            elif buffered_poly.geom_type == 'MultiPolygon':
                global_coastline_polygons.extend(buffered_poly.geoms)

    coastline_tree = STRtree(global_coastline_polygons) if global_coastline_polygons else None
    log.info(f"Successfully loaded and indexed {len(global_coastline_polygons)} static coastlines for collision checking.")
    # -------------------------------------------------------------

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
        if idx1 <= idx0: continue

        w_df = df.iloc[idx0:idx1]
        stats['windows_seen'] += 1

        w_mmsi = w_df['mmsi'].values
        unique_mmsis, inv_indices = np.unique(w_mmsi, return_inverse=True)
        num_unique = len(unique_mmsis)
        if num_unique < 2: continue

        w_lat, w_lon, w_sog = w_df['latitude'].values, w_df['longitude'].values, w_df['sog'].values
        w_cog, w_hdg, w_vtype = w_df['cog'].values, w_df['heading'].values, w_df['vessel_type'].values
        w_len, w_wid, w_draft = w_df['length'].values, w_df['width'].values, w_df['draft'].values
        w_tsec = timestamps_sec[idx0:idx1]

        mmsi_mean_lat = np.zeros(num_unique, dtype=np.float64)
        mmsi_mean_lon = np.zeros(num_unique, dtype=np.float64)
        mmsi_counts = np.bincount(inv_indices, minlength=num_unique)

        np.add.at(mmsi_mean_lat, inv_indices, w_lat)
        np.add.at(mmsi_mean_lon, inv_indices, w_lon)
        mmsi_mean_lat /= mmsi_counts
        mmsi_mean_lon /= mmsi_counts

        coords = np.column_stack((mmsi_mean_lat, mmsi_mean_lon))
        tree = cKDTree(coords)

        use_strict = (np.random.rand() < STRICT_SAMPLING_RATIO) if SAMPLING_MODE == "mixed" else (SAMPLING_MODE == "strict")

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
            if n_large_movers_in_window < TARGET_LARGE_MOVERS_PER_SCENE:
                stats['windows_skipped_insufficient_large_movers'] += 1
                continue
            ego_candidate_mask = large_mover_mask
            if not np.any(ego_candidate_mask):
                stats['skipped_anchor_too_few_rows'] += 1
                continue

            ego_scores = np.zeros(num_unique, dtype=np.float64)
            for i in np.where(ego_candidate_mask)[0]:
                nbrs = tree.query_ball_point(coords[i], r=LOCAL_BOX_DEG, p=np.inf)
                ego_scores[i] = int(np.sum(large_mover_mask[nbrs])) * (1 + densities[i]) * (1 + mmsi_max_len[i] / 100.0)
            anchor_order = np.argsort(-ego_scores[ego_candidate_mask])
            ranked_anchor_indices = np.where(ego_candidate_mask)[0][anchor_order]
        else:
            focus_on_moving_cargo = np.random.rand() < 0.80
            if focus_on_moving_cargo:
                valid_anchor_mask = (densities > 1) & (mmsi_counts >= 2) & (mean_speed_knots > MIN_DYNAMIC_SPEED_KNOTS)
            else:
                valid_anchor_mask = (densities > 1) & (mmsi_counts >= 2)
                
            if not np.any(valid_anchor_mask): continue
                
            anchor_scores = densities[valid_anchor_mask] * (mmsi_max_len[valid_anchor_mask] if focus_on_moving_cargo else 1)
            anchor_order = np.argsort(-anchor_scores)
            ranked_anchor_indices = np.where(valid_anchor_mask)[0][anchor_order]

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
                if mmsi_counts[n_idx] < 2: continue
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

            if anchor_mmsi not in valid_groups or len(valid_groups) < 2:
                stats['skipped_anchor_too_few_rows'] += 1
                continue

            anchor_res = valid_groups[anchor_mmsi]
            if len(anchor_res['t_sec']) < TARGET_FRAMES:
                stats['skipped_too_few_after_resample'] += 1
                continue
                
            best_start_idx = 0
            highest_speed = 0.0
            for i in range(len(anchor_res['t_sec']) - TARGET_FRAMES + 1):
                window_speed = np.mean(anchor_res['sog_ms'][i : i + TARGET_FRAMES])
                if window_speed > highest_speed:
                    highest_speed = window_speed
                    best_start_idx = i
                        
            scene_t_sec = anchor_res['t_sec'][best_start_idx : best_start_idx + TARGET_FRAMES]

            # Drop ships that don't exist during the exact synchronized clock window
            mmsis_to_drop_time = set()
            for m, res in valid_groups.items():
                if not np.all(np.isin(scene_t_sec, res['t_sec'])): mmsis_to_drop_time.add(m)
            for m in mmsis_to_drop_time: del valid_groups[m]
                
            # Here, the anchor MUST exist because it defined the clock window
            if len(valid_groups) < 2 or anchor_mmsi not in valid_groups:
                stats['skipped_temporal_desync'] += 1
                continue

            # Lock the camera origin to the anchor ship
            s_idx_anchor = np.searchsorted(valid_groups[anchor_mmsi]['t_sec'], scene_t_sec[0])
            origin_lat = valid_groups[anchor_mmsi]['latitude'][s_idx_anchor]
            origin_lon = valid_groups[anchor_mmsi]['longitude'][s_idx_anchor]

            grid_key = (round(origin_lat, 2), round(origin_lon, 2))
            if location_usage[grid_key] >= MAX_SCENARIOS_PER_GRID_CELL:
                stats['skipped_location_overused'] += 1
                continue

            if len(valid_groups) > MAX_SHIPS_PER_SCENE:
                stats['boxes_downsampled'] += 1
                ship_priorities = []
                for m in valid_groups:
                    res = valid_groups[m]
                    is_ego = (m == anchor_mmsi)
                    is_large = res['length'] > LARGE_SHIP_LENGTH_M
                    is_cargo_type = is_cargo_or_tanker(res['vessel_type'])
                    mean_sog_knots = np.mean(res['sog_ms']) / 0.514444
                    is_fast_cargo = is_large and is_cargo_type and mean_sog_knots > CARGO_SPEED_KNOTS
                    dist = np.hypot(np.mean(res['latitude']) - origin_lat, np.mean(res['longitude']) - origin_lon)
                    
                    priority = 0 if is_ego else 1 if is_fast_cargo else 2 if is_large else 3 + dist * 1000  
                    ship_priorities.append((priority, dist, m))
                
                ship_priorities.sort(key=lambda x: (x[0], x[1]))
                valid_groups = {m: valid_groups[m] for m in set([x[2] for x in ship_priorities[:MAX_SHIPS_PER_SCENE]])}

            # --- MODIFICATION 1: Pseudo-Ego collision filter (Anchor can be deleted) ---
            valid_groups = filter_cpa_collisions(valid_groups, stats=stats)
            if len(valid_groups) < 2:
                stats['skipped_too_few_after_collision_filter'] += 1
                continue

            # -------------------------------------------------------------
            # GEOMETRIC COASTLINE COLLISION DETECTOR
            # -------------------------------------------------------------
            mmsis_to_drop_coastline = set()
            if coastline_tree is not None:
                for m, res in valid_groups.items():
                    s_idx = np.searchsorted(res['t_sec'], scene_t_sec[0])
                    lons_seq = res['longitude'][s_idx : s_idx + TARGET_FRAMES]
                    lats_seq = res['latitude'][s_idx : s_idx + TARGET_FRAMES]
                    
                    xs, ys = global_transformer.transform(lons_seq, lats_seq)
                    path_line = LineString(np.column_stack((xs, ys)))
                    
                    intersecting_idx = coastline_tree.query(path_line)
                    collision = False
                    for c_idx in intersecting_idx:
                        if global_coastline_polygons[c_idx].intersects(path_line):
                            collision = True
                            break
                            
                    if collision:
                        mmsis_to_drop_coastline.add(m)
                        stats['coastline_collisions_prevented'] += 1
                        
            for m in mmsis_to_drop_coastline: 
                del valid_groups[m]

            # --- MODIFICATION 2: Do NOT drop scenario if anchor gets deleted ---
            if len(valid_groups) < 2:
                stats['skipped_due_to_coastline_collision'] += 1
                continue
            # -------------------------------------------------------------

            dynamic_ship_count, meaningful_movement_count, large_mover_count = 0, 0, 0
            for m, res in valid_groups.items():
                s_idx = np.searchsorted(res['t_sec'], scene_t_sec[0])
                sliced_r = res['yaw_rate_r'][s_idx : s_idx + TARGET_FRAMES]
                sliced_sog = res['sog_ms'][s_idx : s_idx + TARGET_FRAMES]
                
                dx = (res['longitude'][s_idx + TARGET_FRAMES - 1] - res['longitude'][s_idx]) * 111000.0 * np.cos(np.deg2rad(res['latitude'][s_idx]))
                dy = (res['latitude'][s_idx + TARGET_FRAMES - 1] - res['latitude'][s_idx]) * 111000.0
                
                if np.hypot(dx, dy) > MIN_DISPLACEMENT_METERS: meaningful_movement_count += 1
                
                mean_speed_knots = np.mean(sliced_sog) / 0.514444
                total_turn_deg = np.sum(np.abs(sliced_r)) * DT_SECONDS * (180.0 / np.pi)
                
                is_large = res['length'] > LARGE_SHIP_LENGTH_M
                is_cargo_type = is_cargo_or_tanker(res['vessel_type'])
                is_large_cargo_transit = is_large and is_cargo_type and mean_speed_knots > CARGO_SPEED_KNOTS
                
                if is_large_cargo_transit:
                    dynamic_ship_count += 1
                    large_mover_count += 1
                elif mean_speed_knots > MIN_DYNAMIC_SPEED_KNOTS and total_turn_deg > MIN_TURN_DEG_PER_SCENE:
                    dynamic_ship_count += 1
                    
            if meaningful_movement_count < 1 or dynamic_ship_count < (2 if use_strict else 1) or \
               (use_strict and large_mover_count < TARGET_LARGE_MOVERS_PER_SCENE):
                stats['skipped_boring_straight_lines'] += 1
                continue

            current_fleet = set(valid_groups.keys())
            candidate_fleet_ids = set()
            for m in current_fleet: candidate_fleet_ids.update(vessel_to_exported_fleets.get(m, []))
            
            is_redundant = False
            for fid in candidate_fleet_ids:
                past_fleet = exported_scenario_fleets[fid]
                overlap = len(current_fleet & past_fleet)
                if overlap > 0 and (overlap / min(len(current_fleet), len(past_fleet))) > FLEET_OVERLAP_THRESHOLD:
                    is_redundant = True; break
            if is_redundant:
                stats['skipped_redundant_fleet'] += 1
                continue

            # -------------------------------------------------------------
            # RENDER SCENARIO
            # -------------------------------------------------------------
            scenario = Scenario(dt=DT_SECONDS, scenario_id=f"ZAM_Batch-{scenario_counter:04d}_1_T-1")
            transformer = Transformer.from_crs("epsg:4326", f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +units=m", always_xy=True)

            if coastline_tree is not None:
                anchor_x, anchor_y = global_transformer.transform(origin_lon, origin_lat)
                search_box = box(anchor_x - MAX_SCENE_RADIUS_M, anchor_y - MAX_SCENE_RADIUS_M,
                                 anchor_x + MAX_SCENE_RADIUS_M, anchor_y + MAX_SCENE_RADIUS_M)
                
                intersecting_indices = coastline_tree.query(search_box)
                for c_idx in intersecting_indices:
                    poly = global_coastline_polygons[c_idx]
                    local_coords = np.array(poly.exterior.coords) - np.array([anchor_x, anchor_y])
                    scenario.add_objects(StaticObstacle(
                        obstacle_id=scenario.generate_object_id(),
                        obstacle_type=ObstacleType.LAND,
                        obstacle_shape=CRPolygon(local_coords),
                        initial_state=InitialState(position=np.array([0, 0]), orientation=0.0, time_step=0)
                    ))

            n_obstacles_added = 0
            for mmsi_val, res in valid_groups.items():
                s_idx = np.searchsorted(res['t_sec'], scene_t_sec[0])
                state_list = []
                trajectory_is_valid = True
                
                for f_idx in range(TARGET_FRAMES):
                    local_idx = s_idx + f_idx
                    exact_time_step = f_idx  
                    
                    x, y = transformer.transform(res['longitude'][local_idx], res['latitude'][local_idx])
                    if abs(x) > MAX_SCENE_RADIUS_M or abs(y) > MAX_SCENE_RADIUS_M:
                        trajectory_is_valid = False; break
                        
                    heading_rad = np.deg2rad((90.0 - res['heading'][local_idx]) % 360.0)
                    state_list.append(TFState(
                        position=np.array([x, y]), orientation=heading_rad,
                        velocity=res['surge_u'][local_idx], velocity_y=res['sway_v'][local_idx],
                        yaw_rate=res['yaw_rate_r'][local_idx], time_step=exact_time_step
                    ))

                if not trajectory_is_valid or len(state_list) != TARGET_FRAMES: continue
                    
                initial_time_step = state_list[0].time_step
                initial_state = InitialState(
                    position=state_list[0].position, velocity=state_list[0].velocity,
                    velocity_y=state_list[0].velocity_y, orientation=state_list[0].orientation,
                    yaw_rate=state_list[0].yaw_rate, time_step=initial_time_step
                )
                
                shape = Rectangle(length=float(res['length']), width=float(res['width']))
                scenario.add_objects(DynamicObstacle(
                    obstacle_id=scenario.generate_object_id(), obstacle_type=vessel_type_to_obstacle_type(res['vessel_type'], np.mean(res['sog_ms'])),
                    obstacle_shape=shape, initial_state=initial_state,
                    prediction=TrajectoryPrediction(Trajectory(initial_time_step + 1, state_list[1:]), shape),
                    depth=float(res['draft']),
                ))
                n_obstacles_added += 1

            if n_obstacles_added > 1:
                fleet_id = len(exported_scenario_fleets)
                exported_scenario_fleets.append(current_fleet)
                for m in current_fleet: vessel_to_exported_fleets[m].append(fleet_id)

                location_usage[grid_key] += 1
                output_file = os.path.join(output_dir, f"scenario_{scenario_counter:04d}.xml")
                CommonOceanFileWriter(
                    scenario=scenario, planning_problem_set=PlanningProblemSet(),
                    author="Batch ML Pipeline", affiliation="Data Engineering", source="AIS", tags={Tag.OPENSEA}
                ).write_to_file(output_file, OverwriteExistingFile.ALWAYS)

                if on_scenario_created is not None:
                    try:
                        on_scenario_created(output_file)
                    except Exception as err:
                        log.warning("on_scenario_created callback notice: %s", err)

                manifest_entries.append(dict(
                    scenario_id=f"scenario_{scenario_counter:04d}",
                    t_start=str(pd.to_datetime(scene_t_sec[0], unit='s')),
                    t_end=str(pd.to_datetime(scene_t_sec[-1], unit='s')),
                    origin_lat=origin_lat, origin_lon=origin_lon,
                    mmsis=sorted(int(m) for m in current_fleet),
                    n_vessels=n_obstacles_added,
                ))

                # --- MODIFICATION 3: Dynamic Pseudo-Ego Statistics Logging ---
                if anchor_mmsi in valid_groups:
                    pseudo_ego_mmsi = anchor_mmsi
                else:
                    # Original anchor hit land/crashed. Pick largest surviving ship for logs
                    pseudo_ego_mmsi = max(valid_groups.keys(), key=lambda k: valid_groups[k]['length'])
                    
                pseudo_res = valid_groups[pseudo_ego_mmsi]
                anchor_len = pseudo_res['length']
                anchor_mean_sog_knots = np.mean(pseudo_res['sog_ms']) / 0.514444
                
                if (anchor_len > LARGE_SHIP_LENGTH_M and is_cargo_or_tanker(pseudo_res['vessel_type']) and anchor_mean_sog_knots > CARGO_SPEED_KNOTS):
                    stats['scenarios_ego_large'] += 1
                elif anchor_mean_sog_knots > MIN_DYNAMIC_SPEED_KNOTS:
                    stats['scenarios_ego_small'] += 1
                else:
                    stats['scenarios_ego_anchored'] += 1
                
                stats['large_movers_per_scene_sum'] = stats.get('large_movers_per_scene_sum', 0) + large_mover_count
                stats['scenarios_written'] += 1
                
                log_tag = "Pseudo-Ego" if anchor_mmsi not in valid_groups else "Ego"
                log.info("[%d] wrote %s with %d vessels (%s: len=%.0f, sog=%.1fkt, LargeMovers=%d)", 
                         scenario_counter, output_file, n_obstacles_added, log_tag, anchor_len, anchor_mean_sog_knots, large_mover_count)
                scenario_counter += 1

    if manifest_entries:
        with open(manifest_path, "w") as f:
            for e in manifest_entries:
                f.write(json.dumps(e) + "\n")
        log.info("Wrote %d entries to %s", len(manifest_entries), manifest_path)

    _log_summary(stats)
    return stats

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK (Google Drive Integration)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    print("🚀 Starting Local Scenario Generation with Google Drive Coastlines...")
    
    # Target files and directories
    input_zst_file = "/run/media/akshat/Akshat_USB/AISFiles/ais_data_2015_3_11.csv.zst"
    output_directory = "/run/media/akshat/Akshat_USB/generated_scenarios2015_3_11"
    temp_dir = "./Local_Temp_Data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download coastline if not present locally
    COASTLINE_ZIP_ID = "15EklKQ1HhjogCTA4MITfrKKcCwI2WBXp"
    local_zip_path = os.path.join(temp_dir, "coastline-split.zip")
    
    shp_files = glob.glob(os.path.join(temp_dir, "**", "lines.shp"), recursive=True)
    if shp_files:
        local_shp_file = shp_files[0]
        log.info(f"Found existing shapefile at: {local_shp_file}")
    else:
        log.info("Downloading Global Coastline ZIP from Google Drive...")
        import gdown
        gdown.download(id=COASTLINE_ZIP_ID, output=local_zip_path, quiet=False)
        
        log.info("Extracting shapefile...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        if os.path.exists(local_zip_path):
            os.remove(local_zip_path)
            
        shp_files = glob.glob(os.path.join(temp_dir, "**", "lines.shp"), recursive=True)
        if not shp_files:
            log.error("❌ Extracted the ZIP, but could not find 'lines.shp' anywhere inside.")
            sys.exit(1)
        local_shp_file = shp_files[0]
        log.info("Extraction complete. ZIP deleted to save space.")

    # Generate scenarios
    generate_dataset_from_ais(
        zst_filepath=input_zst_file, 
        shp_filepath=local_shp_file, 
        output_dir=output_directory,
        sampling_strategy="density_first"
    )
    
    print("🎉 All processing complete!")