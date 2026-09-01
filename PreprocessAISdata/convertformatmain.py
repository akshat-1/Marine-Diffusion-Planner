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
def download_gdrive_folder(folder_url_or_id: str, target_dir: str) -> list:
    """Downloads all raw AIS dataset files from a Google Drive folder."""
    os.makedirs(target_dir, exist_ok=True)
    folder_id = extract_gdrive_id(folder_url_or_id)
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    log.info(f"📥 Fetching raw AIS files from Google Drive Folder ID: {folder_id}")

    if gdown is None:
        log.error("❌ 'gdown' module is not installed! Run: pip install gdown")
        return []

    try:
        downloaded = gdown.download_folder(url=folder_url, output=target_dir, quiet=False, remaining_ok=True)
        if downloaded:
            log.info(f"✅ Successfully downloaded {len(downloaded)} raw files into {target_dir}")
            return downloaded
    except Exception as e:
        log.warning(f"⚠️ gdown folder download notice: {e}")

    # Search for files already downloaded in target_dir or subfolders
    local_files = []
    for ext in ['*.csv', '*.csv.zip', '*.zip', '*.gz', '*.zst']:
        local_files.extend(glob.glob(os.path.join(target_dir, '**', ext), recursive=True))

    log.info(f"📁 Local raw file pool in '{target_dir}': {len(local_files)} files found.")
    return local_files


def get_gdrive_api_service(credentials_path=None):
    """Initializes Google Drive API v3 client if credentials JSON is available."""
    if build is None:
        log.warning("⚠️ google-api-python-client is not installed.")
        return None

    SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
    creds = None

    # 1. Search for credentials file
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
        # Check if service account JSON
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


def upload_processed_to_gdrive(local_file_path: str, output_folder_id_or_url: str, credentials_path=None):
    """Uploads a converted .csv.zst dataset file to the specified Google Drive destination folder."""
    output_folder_id = extract_gdrive_id(output_folder_id_or_url)
    filename = os.path.basename(local_file_path)
    log.info(f"\n📤 Uploading {filename} to Google Drive Folder: {output_folder_id}...")

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
            log.info(f"✅ Successfully uploaded to Google Drive via API!")
            log.info(f"   File ID: {uploaded_file.get('id')}")
            log.info(f"   Link: {uploaded_file.get('webViewLink')}")
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

    # Method 3: Informative instructions for uploading via browser/rclone
    log.warning(f"\nℹ️ Automated OAuth upload requires Google Drive API credentials.")
    log.info(f"   Target Folder: https://drive.google.com/drive/folders/{output_folder_id}")
    log.info(f"   Local Processed File: {os.path.abspath(local_file_path)}")
    log.info(f"   To enable 1-click automatic upload: save your Google OAuth/Service Account JSON to 'credentials.json'.\n")
    return False


# ---------------------------------------------------------------------------
# Complete Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_gdrive_ais_pipeline(
    raw_gdrive_url=DEFAULT_RAW_GDRIVE_URL,
    output_gdrive_url=DEFAULT_CONVERTED_GDRIVE_URL,
    local_raw_dir="./raw_ais_downloads",
    local_processed_dir="./processed_ais_outputs",
    credentials_path=None
):
    """Fetches raw AIS files from Google Drive, preprocesses them, and uploads converted datasets."""
    log.info("=========================================================================")
    log.info("🚢 MARITIME AIS DATASET PIPELINE: GOOGLE DRIVE FETCH, CONVERT & UPLOAD")
    log.info("=========================================================================")
    log.info(f"Input Google Drive Folder:  {raw_gdrive_url}")
    log.info(f"Output Google Drive Folder: {output_gdrive_url}")
    log.info(f"Local Raw Directory:        {local_raw_dir}")
    log.info(f"Local Processed Directory:  {local_processed_dir}")
    log.info("=========================================================================\n")

    # Step 1: Download raw AIS data files
    raw_files = download_gdrive_folder(raw_gdrive_url, local_raw_dir)
    if not raw_files:
        log.warning(f"⚠️ No raw AIS files found in '{local_raw_dir}'. Checking for existing local files...")
        raw_files = glob.glob(os.path.join(local_raw_dir, "*.csv")) + glob.glob(os.path.join(local_raw_dir, "*.zip"))

    if not raw_files:
        log.error(f"❌ No raw files available to process. Please place raw .csv files in '{local_raw_dir}'.")
        return

    # Step 2: Preprocess each raw file & upload to Google Drive
    for raw_file in raw_files:
        basename = os.path.basename(raw_file)
        clean_name = re.sub(r'\.(csv|zip|gz)$', '', basename, flags=re.IGNORECASE) + ".csv.zst"
        output_zst_path = os.path.join(local_processed_dir, clean_name)

        # Process file
        processed_path = format_noaa_dataset(raw_file, output_zst_path)

        # Upload to Google Drive target folder
        if processed_path and os.path.exists(processed_path):
            upload_processed_to_gdrive(processed_path, output_gdrive_url, credentials_path)

    os.system('sync')
    log.info("\n🎉 All AIS scenarios fetched, converted, and processed successfully!")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch, Preprocess, and Upload AIS Data to Google Drive.")
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
