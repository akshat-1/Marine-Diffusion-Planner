import os
import sys
import json
import logging
import glob
import gc
import re
import pyarrow.csv as pv
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Force console logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
log = logging.getLogger("ais_formatter")

# ---------------------------------------------------------------------------
# "Time-Travel" AIS Configuration Engine
# ---------------------------------------------------------------------------
def detect_dataset_year(filepath):
    """Extracts the 4-digit year from the filename (e.g., 'ais-2026-01-01.csv' -> 2026)."""
    filename = os.path.basename(filepath)
    match = re.search(r'(20\d{2})', filename)
    if match:
        return int(match.group(1))
    log.warning(f"⚠️ Could not detect year in filename '{filename}'. Defaulting to latest schema (2025+).")
    return 2025

def get_year_specific_rules(year):
    """Returns the exact column mappings and mathematical cleaning rules for the given era."""
    
    # 1. Base Aggressive Column Normalizer (strips spaces and underscores)
    # This funnel catches both the pre-2025 "LAT/LON" and post-2025 swapped "longitude/latitude" names.
    col_mapping = {
        'mmsi': 'mmsi',
        'basedatetime': 'base_date_time',
        'timestamp': 'base_date_time',
        'time': 'base_date_time',
        'lat': 'latitude',
        'latitude': 'latitude',
        'lon': 'longitude',
        'longitude': 'longitude',
        'sog': 'sog',
        'speed': 'sog',
        'cog': 'cog',
        'course': 'cog',
        'heading': 'heading',
        'hdg': 'heading',
        'vesseltype': 'vessel_type',
        'shiptype': 'vessel_type',
        'length': 'length',
        'width': 'width',
        'beam': 'width',  # Some older datasets used Beam
        'draft': 'draft',
        'transceiverclass': 'transceiver_class',
        'class': 'transceiver_class', # Caught post-2025 shortened names
        'transclass': 'transceiver_class'
    }

    # 2. Year-specific Sentinel Value Rules
    rules = {
        "mapping": col_mapping,
        "has_sentinels": True,    # Do we need to mathematically scrub 511s and 102.3s?
        "float_dims": False,      # Are length/width decimals (pre-2018) or ints?
        "has_class_b": True       # Are small vessels included?
    }

    if year >= 2025:
        log.info(f"📅 Detected Era: 2025+ (Modern Schema). Disabling Sentinel scrubbers, expecting swapped coordinates.")
        rules["has_sentinels"] = False # 2025+ uses pure Nulls
    elif year >= 2018:
        log.info(f"📅 Detected Era: 2018-2024. Enabling Sentinel scrubbers (511, 102.3), expecting Class B vessels.")
    else:
        log.info(f"📅 Detected Era: Pre-2018. Enabling float-dimension rules, expecting mostly Class A commercial vessels.")
        rules["float_dims"] = True
        rules["has_class_b"] = False

    return rules

# ---------------------------------------------------------------------------
# High-Speed PyArrow File Formatter
# ---------------------------------------------------------------------------
def format_noaa_dataset(input_csv, output_zst):
    log.info(f"\n🚀 Processing: {input_csv}")
    
    # 1. Detect Year and Rules
    year = detect_dataset_year(input_csv)
    rules = get_year_specific_rules(year)
    
    def normalize_col(c):
        """Strips everything except a-z and 0-9 to catch sneaky formatting changes."""
        return re.sub(r'[^a-z0-9]', '', str(c).lower())

    # 2. PyArrow High-Speed Read
    try:
        table = pv.read_csv(
            input_csv,
            convert_options=pv.ConvertOptions(
                column_types={'MMSI': 'int32', 'mmsi': 'int32'}
            )
        )
        df = table.to_pandas()
    except Exception as e:
        log.error(f"❌ PyArrow failed to read {input_csv}: {e}")
        return
    
    log.info(f"Loaded {len(df)} raw rows into RAM.")

    # 3. Dynamic Schema Application
    df.columns = [normalize_col(c) for c in df.columns]
    
    # Drop columns we don't care about to save RAM instantly
    cols_to_keep = [c for c in df.columns if c in rules["mapping"]]
    df = df[cols_to_keep]
    df = df.rename(columns=rules["mapping"])

    # --- FAILSAFE: Inject missing core columns ---
    expected_cols = ['mmsi', 'base_date_time', 'latitude', 'longitude', 'sog', 'cog', 'heading', 'vessel_type', 'length', 'width', 'draft']
    for col in expected_cols:
        if col not in df.columns:
            log.warning(f"⚠️ Column '{col}' missing from {year} dataset. Injecting blanks to prevent crashes.")
            df[col] = np.nan

    # 4. Type Coercion and Data Cleaning
    log.info("Sanitizing coordinates, dates, and types...")
    df['base_date_time'] = pd.to_datetime(df['base_date_time'], errors='coerce')
    
    for col in ['mmsi', 'vessel_type', 'latitude', 'longitude', 'sog', 'cog', 'heading', 'length', 'width', 'draft']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop physically broken rows (Missing GPS or Date)
    initial_len = len(df)
    df = df.dropna(subset=['latitude', 'longitude', 'base_date_time', 'mmsi'])
    log.info(f"Dropped {initial_len - len(df)} rows due to missing critical GPS/Time data.")

    if len(df) == 0:
        log.error("❌ CRITICAL: No valid rows left after coordinate cleaning!")
        return

    # 5. Era-Specific Mathematics (The Sentinel Scrubber)
    if rules["has_sentinels"]:
        log.info("Scrubbing pre-2025 USCG Sentinel values (511, 102.3, 360.0)...")
        # Replace USCG specific "not available" sentinels with actual NaN
        df.loc[df['heading'] == 511, 'heading'] = np.nan
        df.loc[df['sog'] >= 102.3, 'sog'] = np.nan
        df.loc[df['cog'] >= 360.0, 'cog'] = np.nan

    # Zero-dimension cleanup (Common across all eras)
    df.loc[df['length'] <= 0, 'length'] = np.nan
    df.loc[df['width'] <= 0, 'width'] = np.nan
    df.loc[df['draft'] <= 0, 'draft'] = np.nan

    # Downcast memory footprint
    df['mmsi'] = df['mmsi'].astype(np.int64)
    df['vessel_type'] = df['vessel_type'].fillna(-1).astype(np.int32)
    
    if not rules["float_dims"]:
        # Post-2018 datasets use integer dimensions. Downcasting saves RAM.
        for col in ['length', 'width']:
            df[col] = df[col].fillna(0).astype(np.int32)

    # 6. Save standard ZST
    os.makedirs(os.path.dirname(output_zst), exist_ok=True)
    log.info(f"Saving {len(df)} perfectly standardized rows to {output_zst}...")
    df.to_csv(output_zst, index=False, compression='zstd')
    log.info(f"✅ {os.path.basename(output_zst)} is ready for the CommonRoad Pipeline.")

# ---------------------------------------------------------------------------
# Multiprocessing Main Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage for a single file:
    local_output = "/run/media/akshat/Akshat_USB/AISFiles/ais_data_2015_3_11.csv.zst"
    format_noaa_dataset("/run/media/akshat/Akshat_USB/ais-2025-03-11.csv", local_output)
    
    os.system('sync')
    log.info("Finished.")