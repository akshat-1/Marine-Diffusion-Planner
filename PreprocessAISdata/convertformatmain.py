import os
import sys
import json
import logging
import glob
import gc
import re
import argparse
import pyarrow.csv as pv
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    import gdown
except ImportError:
    gdown = None

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError:
    build = None

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    GoogleDrive = None

# Force console logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
log = logging.getLogger("ais_formatter")

# ---------------------------------------------------------------------------
# Default Google Drive Shared Folder URLs / IDs
# ---------------------------------------------------------------------------
DEFAULT_RAW_GDRIVE_URL = "https://drive.google.com/drive/folders/1lNzEXWGFiOmJbXxfNqXEcPrz_HOD2jYl?usp=sharing"
DEFAULT_CONVERTED_GDRIVE_URL = "https://drive.google.com/drive/folders/1KVs4XuUANFNscoFWhJImEHtUVSwA3LLs?usp=sharing"


def extract_gdrive_id(url_or_id: str) -> str:
    """Extracts a Google Drive folder or file ID from a URL or returns the string if already an ID."""
    if not url_or_id:
        return ""
    url_or_id = url_or_id.strip()
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', url_or_id) or re.search(r'id=([a-zA-Z0-9_-]+)', url_or_id) or re.search(r'/d/([a-zA-Z0-9_-]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id


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
        'beam': 'width',
        'draft': 'draft',
        'transceiverclass': 'transceiver_class',
        'class': 'transceiver_class',
        'transclass': 'transceiver_class'
    }

    rules = {
        "mapping": col_mapping,
        "has_sentinels": True,
        "float_dims": False,
        "has_class_b": True
    }

    if year >= 2025:
        log.info(f"📅 Detected Era: 2025+ (Modern Schema). Disabling Sentinel scrubbers, expecting swapped coordinates.")
        rules["has_sentinels"] = False
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
    """Preprocesses a raw AIS CSV file into standardized, ZST-compressed format."""
    log.info(f"\n🚀 Processing: {input_csv}")
    
    if not os.path.exists(input_csv) or os.path.getsize(input_csv) == 0:
        log.warning(f"⚠️ Skipping empty or non-existent file: {input_csv}")
        return None

    year = detect_dataset_year(input_csv)
    rules = get_year_specific_rules(year)
    
    def normalize_col(c):
        return re.sub(r'[^a-z0-9]', '', str(c).lower())

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
        return None
    
    log.info(f"Loaded {len(df)} raw rows into RAM.")

    df.columns = [normalize_col(c) for c in df.columns]
    
    cols_to_keep = [c for c in df.columns if c in rules["mapping"]]
    df = df[cols_to_keep]
    df = df.rename(columns=rules["mapping"])

    expected_cols = ['mmsi', 'base_date_time', 'latitude', 'longitude', 'sog', 'cog', 'heading', 'vessel_type', 'length', 'width', 'draft']
    for col in expected_cols:
        if col not in df.columns:
            log.warning(f"⚠️ Column '{col}' missing from {year} dataset. Injecting blanks to prevent crashes.")
            df[col] = np.nan

    log.info("Sanitizing coordinates, dates, and types...")
    df['base_date_time'] = pd.to_datetime(df['base_date_time'], errors='coerce')
    
    for col in ['mmsi', 'vessel_type', 'latitude', 'longitude', 'sog', 'cog', 'heading', 'length', 'width', 'draft']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_len = len(df)
    df = df.dropna(subset=['latitude', 'longitude', 'base_date_time', 'mmsi'])
    log.info(f"Dropped {initial_len - len(df)} rows due to missing critical GPS/Time data.")

    if len(df) == 0:
        log.error("❌ CRITICAL: No valid rows left after coordinate cleaning!")
        return None

    if rules["has_sentinels"]:
        log.info("Scrubbing pre-2025 USCG Sentinel values (511, 102.3, 360.0)...")
        df.loc[df['heading'] == 511, 'heading'] = np.nan
        df.loc[df['sog'] >= 102.3, 'sog'] = np.nan
        df.loc[df['cog'] >= 360.0, 'cog'] = np.nan

    df.loc[df['length'] <= 0, 'length'] = np.nan
    df.loc[df['width'] <= 0, 'width'] = np.nan
    df.loc[df['draft'] <= 0, 'draft'] = np.nan

    df['mmsi'] = df['mmsi'].astype(np.int64)
    df['vessel_type'] = df['vessel_type'].fillna(-1).astype(np.int32)
    
    if not rules["float_dims"]:
        for col in ['length', 'width']:
            df[col] = df[col].fillna(0).astype(np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(output_zst)), exist_ok=True)
    log.info(f"Saving {len(df)} perfectly standardized rows to {output_zst}...")
    df.to_csv(output_zst, index=False, compression='zstd')
    log.info(f"✅ {os.path.basename(output_zst)} is ready for the CommonRoad Pipeline.")
    return output_zst


# ---------------------------------------------------------------------------
# Google Drive Download & Upload Engine
# ---------------------------------------------------------------------------
def get_gdrive_api_service(credentials_path=None):
    """Initializes Google Drive API v3 client if credentials JSON is available."""
    if build is None:
        log.warning("⚠️ google-api-python-client is not installed.")
        return None

    SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
    creds = None

    possible_cred_files = [
        credentials_path,
        "credentials.json",
        "service_account.json",
        "client_secrets.json",
        "token.json",
        os.path.expanduser("~/.config/gdrive/credentials.json")
    ]
    
    cred_file = None
    for p in possible_cred_files:
        if p and os.path.exists(p):
            cred_file = p
            break

    if not cred_file:
        return None

    try:
        with open(cred_file, 'r') as f:
            data = json.load(f)
        if data.get('type') == 'service_account':
            creds = service_account.Credentials.from_service_account_file(cred_file, scopes=SCOPES)
            log.info(f"🔑 Authenticated using Google Service Account: {data.get('client_email', cred_file)}")
        else:
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
                    creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            log.info("🔑 Authenticated using Google OAuth2 User Credentials.")
        
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        log.warning(f"⚠️ Google API authentication notice ({cred_file}): {e}")
        return None


def get_gdrive_folder_file_list(folder_url_or_id: str, credentials_path=None) -> list:
    """
    Retrieves file metadata list from a Google Drive folder without downloading content to disk.
    Returns: [{'id': file_id, 'name': filename}, ...]
    """
    folder_id = extract_gdrive_id(folder_url_or_id)
    items = []

    # Method 1: Google Drive API v3
    service = get_gdrive_api_service(credentials_path)
    if service is not None:
        try:
            page_token = None
            while True:
                response = service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    spaces='drive',
                    fields='nextPageToken, files(id, name, size)',
                    pageToken=page_token
                ).execute()
                for file in response.get('files', []):
                    items.append({'id': file.get('id'), 'name': file.get('name')})
                page_token = response.get('nextPageToken', None)
                if not page_token:
                    break
            if items:
                log.info(f"📋 Retrieved {len(items)} file metadata entries via Google Drive API.")
                return items
        except Exception as e:
            log.warning(f"⚠️ Google Drive API file listing notice: {e}")

    # Method 2: gdown folder listing (skip_download=True)
    if gdown is not None:
        try:
            res = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
            if res:
                for obj in res:
                    if hasattr(obj, 'id') and hasattr(obj, 'path'):
                        items.append({'id': obj.id, 'name': os.path.basename(obj.path)})
                    elif isinstance(obj, dict):
                        items.append({'id': obj.get('id'), 'name': obj.get('name')})
            if items:
                log.info(f"📋 Retrieved {len(items)} file metadata entries via gdown folder scanner.")
                return items
        except Exception as e:
            log.warning(f"⚠️ gdown file listing notice: {e}")

    return items


def upload_processed_to_gdrive(local_file_path: str, output_folder_id_or_url: str, credentials_path=None):
    """Uploads a converted .csv.zst dataset file to the specified Google Drive destination folder."""
    output_folder_id = extract_gdrive_id(output_folder_id_or_url)
    filename = os.path.basename(local_file_path)
    log.info(f"📤 Uploading {filename} to Google Drive Folder: {output_folder_id}...")

    # Method 1: Google Drive v3 API
    service = get_gdrive_api_service(credentials_path)
    if service is not None:
        try:
            file_metadata = {
                'name': filename,
                'parents': [output_folder_id]
            }
            media = MediaFileUpload(local_file_path, mimetype='application/zstd', resumable=True)
            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()
            log.info(f"✅ Successfully uploaded via Google Drive API! File ID: {uploaded_file.get('id')}")
            return True
        except Exception as e:
            log.error(f"❌ Google Drive API upload failed: {e}")

    # Method 2: PyDrive2 fallback
    if GoogleDrive is not None:
        try:
            gauth = GoogleAuth()
            gauth.LoadCredentialsFile("mycreds.txt")
            if gauth.credentials is None:
                gauth.CommandLineAuth()
            elif gauth.access_token_expired:
                gauth.Refresh()
            else:
                gauth.Authorize()
            gauth.SaveCredentialsFile("mycreds.txt")
            
            drive = GoogleDrive(gauth)
            drive_file = drive.CreateFile({
                'title': filename,
                'parents': [{'id': output_folder_id}]
            })
            drive_file.SetContentFile(local_file_path)
            drive_file.Upload()
            log.info(f"✅ Successfully uploaded via PyDrive2! File ID: {drive_file.get('id')}")
            return True
        except Exception as e:
            log.warning(f"⚠️ PyDrive2 upload notice: {e}")

    log.warning(f"ℹ️ Automated OAuth upload requires Google Drive API credentials ('credentials.json').")
    log.info(f"   Target Folder: https://drive.google.com/drive/folders/{output_folder_id}")
    return False


# ---------------------------------------------------------------------------
# Streaming Single-File Pipeline Orchestrator (0-Disk Accumulation)
# ---------------------------------------------------------------------------
def run_gdrive_ais_pipeline(
    raw_gdrive_url=DEFAULT_RAW_GDRIVE_URL,
    output_gdrive_url=DEFAULT_CONVERTED_GDRIVE_URL,
    local_raw_dir="./raw_ais_downloads",
    local_processed_dir="./processed_ais_outputs",
    credentials_path=None
):
    """
    Streaming Single-File AIS Pipeline (Zero Disk Accumulation):
    Iterates file-by-file through the raw Google Drive folder:
      1. Downloads ONLY the single raw file to local storage.
      2. Preprocesses it via format_noaa_dataset().
      3. Uploads converted .csv.zst to destination Google Drive folder.
      4. Immediately deletes local raw and processed files to free up disk space.
    """
    log.info("=========================================================================")
    log.info("🚢 MARITIME AIS DATASET PIPELINE: STREAMING SINGLE-FILE FETCH & CONVERT")
    log.info("=========================================================================")
    log.info(f"Input Google Drive Folder:  {raw_gdrive_url}")
    log.info(f"Output Google Drive Folder: {output_gdrive_url}")
    log.info("Strategy:                   0-Disk-Accumulation Single-File Streaming")
    log.info("=========================================================================\n")

    os.makedirs(local_raw_dir, exist_ok=True)
    os.makedirs(local_processed_dir, exist_ok=True)

    # Step 1: List all files in Google Drive raw input folder without downloading content
    file_items = get_gdrive_folder_file_list(raw_gdrive_url, credentials_path)

    if not file_items:
        log.warning(f"⚠️ Could not list remote folder files online. Checking local file pool in '{local_raw_dir}'...")
        local_existing = glob.glob(os.path.join(local_raw_dir, "*"))
        file_items = [{'id': None, 'name': os.path.basename(f), 'local_path': f} for f in local_existing if os.path.isfile(f)]

    if not file_items:
        log.error(f"❌ No raw files available to process.")
        return

    total_files = len(file_items)
    log.info(f"🔄 Starting streaming pipeline for {total_files} total raw files...\n")

    for idx, item in enumerate(file_items, 1):
        file_id = item.get('id')
        filename = item.get('name', f"raw_ais_{idx}.csv")
        log.info("-------------------------------------------------------------------------")
        log.info(f"📦 [{idx}/{total_files}] Processing file: {filename}")
        log.info("-------------------------------------------------------------------------")

        local_raw_path = item.get('local_path') or os.path.join(local_raw_dir, filename)

        # 1. Download single file if not locally present
        if file_id and not os.path.exists(local_raw_path):
            log.info(f"📥 Downloading single raw file [{idx}/{total_files}]: {filename}...")
            try:
                gdown.download(id=file_id, output=local_raw_path, quiet=False)
            except Exception as e:
                log.error(f"❌ Failed to download {filename} (ID: {file_id}): {e}")
                continue

        if not os.path.exists(local_raw_path) or os.path.getsize(local_raw_path) == 0:
            log.warning(f"⚠️ Skipping 0-byte or unreadable file: {filename}")
            if os.path.exists(local_raw_path):
                try:
                    os.remove(local_raw_path)
                except Exception:
                    pass
            continue

        # 2. Preprocess single file
        clean_name = re.sub(r'\.(csv|zip|gz)$', '', filename, flags=re.IGNORECASE) + ".csv.zst"
        local_processed_path = os.path.join(local_processed_dir, clean_name)

        processed_path = format_noaa_dataset(local_raw_path, local_processed_path)

        # 3. Upload converted file to destination Google Drive folder
        if processed_path and os.path.exists(processed_path):
            upload_processed_to_gdrive(processed_path, output_gdrive_url, credentials_path)

        # 4. Immediate Cleanup: Free up disk space before moving to next file!
        log.info(f"🧹 Freeing up disk space for [{idx}/{total_files}]...")
        if os.path.exists(local_raw_path):
            try:
                os.remove(local_raw_path)
                log.info(f"   Deleted local raw file: {os.path.basename(local_raw_path)}")
            except Exception as e:
                log.warning(f"   Could not remove {local_raw_path}: {e}")

        if processed_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
                log.info(f"   Deleted local processed file: {os.path.basename(processed_path)}")
            except Exception as e:
                log.warning(f"   Could not remove {processed_path}: {e}")

        gc.collect()
        log.info(f"✨ [{idx}/{total_files}] Complete! Disk space clean.\n")

    os.system('sync')
    log.info("🎉 All raw AIS files processed, converted, uploaded, and disk space freed up successfully!")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Single-File Fetch, Preprocess, and Upload AIS Data to Google Drive.")
    parser.add_argument("--gdrive-input", type=str, default=DEFAULT_RAW_GDRIVE_URL, help="Input Google Drive Folder URL or ID")
    parser.add_argument("--gdrive-output", type=str, default=DEFAULT_CONVERTED_GDRIVE_URL, help="Output Google Drive Folder URL or ID")
    parser.add_argument("--raw-dir", type=str, default="./raw_ais_downloads", help="Local directory to store raw downloaded AIS files")
    parser.add_argument("--processed-dir", type=str, default="./processed_ais_outputs", help="Local directory to store converted .csv.zst files")
    parser.add_argument("--credentials", type=str, default=None, help="Path to Google Drive OAuth2 / Service Account credentials JSON")
    parser.add_argument("--local-input", type=str, default=None, help="Path to a single local raw CSV file to format directly")
    parser.add_argument("--local-output", type=str, default=None, help="Path to output converted .csv.zst file")

    args = parser.parse_args()

    if args.local_input:
        out_path = args.local_output or (args.local_input + ".zst")
        format_noaa_dataset(args.local_input, out_path)
    else:
        run_gdrive_ais_pipeline(
            raw_gdrive_url=args.gdrive_input,
            output_gdrive_url=args.gdrive_output,
            local_raw_dir=args.raw_dir,
            local_processed_dir=args.processed_dir,
            credentials_path=args.credentials
        )
