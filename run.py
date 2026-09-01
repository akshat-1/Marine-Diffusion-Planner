#!/usr/bin/env python3
"""
Master Maritime AIS End-to-End Pipeline Orchestrator (run.py)

Connects:
  1. PreprocessAISdata/convertformatmain.py (Data sanitization & format conversion)
  2. SceneriosGenerator/aisdatageneratoroffline.py (CommonOcean XML Scenario Generation)

Pipeline Architecture (0-Disk-Accumulation Multi-Threaded Streaming):
  - Uses NUM_THREADS (default = 3, configurable via --num-threads)
  - Fetches remote raw AIS file list from Google Drive (Folder ID: 1lNzEXWGFiOmJbXxfNqXEcPrz_HOD2jYl)
  - For each raw AIS file in the queue, a thread performs:
      Step 1: Download ONLY 1 raw AIS file from Google Drive.
      Step 2: Preprocess raw file via format_noaa_dataset() -> clean .csv.zst.
      Step 3: IMMEDIATELY REMOVE downloaded raw file from local disk.
      Step 4: Run generate_dataset_from_ais() -> CommonOcean .xml scenario files.
      Step 5: IMMEDIATELY REMOVE formatted .csv.zst file from local disk.
      Step 6: Upload generated CommonOcean .xml scenario files to Google Drive
              (Folder ID: 1ZwarzUx9ieYriC81-ryMIppdZa3Sucvd).
      Step 7: IMMEDIATELY REMOVE generated .xml scenario files from local disk.
      Step 8: Move to the next raw file in the queue.
"""

import os
import sys
import glob
import re
import gc
import shutil
import time
import json
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Ensure repository root & subpackages are in Python import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from PreprocessAISdata.convertformatmain import (
    format_noaa_dataset,
    get_gdrive_folder_file_list,
    upload_processed_to_gdrive,
    extract_gdrive_id,
    get_gdrive_api_service
)
from SceneriosGenerator.aisdatageneratoroffline import generate_dataset_from_ais

try:
    import gdown
except ImportError:
    gdown = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# Default Configuration Constants & Folder URLs
# ---------------------------------------------------------------------------
DEFAULT_NUM_THREADS = 3  # Default 3 worker threads as requested
DEFAULT_RAW_GDRIVE_URL = "https://drive.google.com/drive/folders/1lNzEXWGFiOmJbXxfNqXEcPrz_HOD2jYl?usp=sharing"
DEFAULT_SCENARIOS_GDRIVE_URL = "https://drive.google.com/drive/folders/1ZwarzUx9ieYriC81-ryMIppdZa3Sucvd?usp=sharing"

LOCAL_TEMP_DIR = os.path.join(SCRIPT_DIR, "Local_Temp_Data")
LOCAL_RAW_DOWNLOADS_DIR = os.path.join(SCRIPT_DIR, "raw_ais_downloads")
LOCAL_FORMATTED_DIR = os.path.join(SCRIPT_DIR, "formatted_zst_temp")
LOCAL_SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "generated_scenarios_temp")


def ensure_coastline_shapefile() -> str:
    """Ensures global coastline shapefile is downloaded and available locally."""
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    shp_files = glob.glob(os.path.join(LOCAL_TEMP_DIR, "**", "lines.shp"), recursive=True)
    if shp_files:
        log.info(f"🗺️ Found existing coastline shapefile at: {shp_files[0]}")
        return shp_files[0]

    log.info("🗺️ Global coastline shapefile not found locally. Downloading from Google Drive...")
    COASTLINE_ZIP_ID = "15EklKQ1HhjogCTA4MITfrKKcCwI2WBXp"
    local_zip_path = os.path.join(LOCAL_TEMP_DIR, "coastline-split.zip")

    if gdown is None:
        raise RuntimeError("gdown module is required to download coastline shapefile. Run: pip install gdown")

    gdown.download(id=COASTLINE_ZIP_ID, output=local_zip_path, quiet=False)

    log.info("📦 Extracting coastline shapefile...")
    import zipfile
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_TEMP_DIR)

    if os.path.exists(local_zip_path):
        os.remove(local_zip_path)

    shp_files = glob.glob(os.path.join(LOCAL_TEMP_DIR, "**", "lines.shp"), recursive=True)
    if not shp_files:
        raise FileNotFoundError("Extracted coastline ZIP, but could not locate 'lines.shp'.")

    log.info(f"✅ Coastline shapefile ready at: {shp_files[0]}")
    return shp_files[0]


# ---------------------------------------------------------------------------
# Thread Worker Pipeline Function
# ---------------------------------------------------------------------------
def process_file_worker(
    item: dict,
    idx: int,
    total_files: int,
    shp_filepath: str,
    raw_gdrive_url: str,
    scenarios_gdrive_url: str,
    credentials_path: str = None
) -> bool:
    """
    Worker function executed by worker threads:
      Step 1: Download single raw AIS file from Google Drive.
      Step 2: Preprocess raw file via convertformatmain.py -> clean .csv.zst.
      Step 3: Immediately remove downloaded raw file from local disk.
      Step 4: Run aisdatageneratoroffline.py -> generate CommonOcean .xml scenario files.
      Step 5: Immediately remove formatted .csv.zst file from local disk.
      Step 6: Upload generated .xml scenario files to Google Drive output folder.
      Step 7: Immediately remove generated .xml scenario files from local disk.
      Step 8: Thread completes item and moves to next available file in queue.
    """
    thread_name = threading.current_thread().name
    file_id = item.get('id')
    filename = item.get('name', f"raw_ais_{idx}.csv")
    
    # Thread-specific staging directories to prevent file collisions
    thread_id_str = f"thread_{idx}"
    thread_raw_dir = os.path.join(LOCAL_RAW_DOWNLOADS_DIR, thread_id_str)
    thread_formatted_dir = os.path.join(LOCAL_FORMATTED_DIR, thread_id_str)
    thread_scenarios_dir = os.path.join(LOCAL_SCENARIOS_DIR, thread_id_str)

    os.makedirs(thread_raw_dir, exist_ok=True)
    os.makedirs(thread_formatted_dir, exist_ok=True)
    os.makedirs(thread_scenarios_dir, exist_ok=True)

    local_raw_path = item.get('local_path') or os.path.join(thread_raw_dir, filename)

    log.info("=========================================================================")
    log.info(f"🚀 [{thread_name}] Starting File [{idx}/{total_files}]: {filename}")
    log.info("=========================================================================")

    # -----------------------------------------------------------------------
    # Step 1: Download ONLY 1 raw AIS file from Google Drive
    # -----------------------------------------------------------------------
    if file_id and not os.path.exists(local_raw_path):
        log.info(f"📥 [{thread_name}] Step 1/7: Downloading raw file [{idx}/{total_files}]: {filename}...")
        try:
            dl_start = time.time()
            gdown.download(id=file_id, output=local_raw_path, quiet=True)
            dl_bytes = os.path.getsize(local_raw_path) if os.path.exists(local_raw_path) else 0
            log.info(f"📥 [{thread_name}] Downloaded {filename} in {time.time() - dl_start:.2f}s ({dl_bytes / (1024*1024):.2f} MB)")
        except Exception as e:
            log.error(f"❌ [{thread_name}] Failed to download {filename} (ID: {file_id}): {e}")
            return False

    if not os.path.exists(local_raw_path) or os.path.getsize(local_raw_path) == 0:
        log.warning(f"⚠️ [{thread_name}] Skipping 0-byte or unreadable raw file: {filename}")
        if os.path.exists(local_raw_path):
            try:
                os.remove(local_raw_path)
            except Exception:
                pass
        return False

    raw_bytes = os.path.getsize(local_raw_path)

    # -----------------------------------------------------------------------
    # Step 2: Format & Preprocess Raw File via convertformatmain.py
    # -----------------------------------------------------------------------
    clean_base = re.sub(r'\.(csv|zip|gz|zst|zstd)+$', '', filename, flags=re.IGNORECASE)
    clean_base = re.sub(r'\.csv$', '', clean_base, flags=re.IGNORECASE)
    formatted_zst_name = clean_base + ".csv.zst"
    formatted_zst_path = os.path.join(thread_formatted_dir, formatted_zst_name)

    log.info(f"⚙️ [{thread_name}] Step 2/7: Preprocessing & formatting raw data $\\rightarrow$ {formatted_zst_name}...")
    formatted_path = format_noaa_dataset(local_raw_path, formatted_zst_path)

    # -----------------------------------------------------------------------
    # Step 3: IMMEDIATELY REMOVE Downloaded Raw File from Local Disk
    # -----------------------------------------------------------------------
    log.info(f"🧹 [{thread_name}] Step 3/7: Removing raw downloaded file to free disk space ({raw_bytes / (1024*1024):.2f} MB)...")
    if os.path.exists(local_raw_path):
        try:
            os.remove(local_raw_path)
            log.info(f"   Deleted raw file: {os.path.basename(local_raw_path)}")
        except Exception as e:
            log.warning(f"   Could not remove {local_raw_path}: {e}")

    if not formatted_path or not os.path.exists(formatted_path):
        log.error(f"❌ [{thread_name}] Preprocessing failed for {filename}. Skipping scenario generation.")
        return False

    formatted_bytes = os.path.getsize(formatted_path)

    # -----------------------------------------------------------------------
    # Step 4: Generate CommonOcean XML Scenarios via aisdatageneratoroffline.py
    # -----------------------------------------------------------------------
    log.info(f"🚢 [{thread_name}] Step 4/7: Generating CommonOcean XML Scenarios from {formatted_zst_name}...")
    try:
        gen_start = time.time()
        stats = generate_dataset_from_ais(
            zst_filepath=formatted_path,
            shp_filepath=shp_filepath,
            output_dir=thread_scenarios_dir,
            sampling_strategy="density_first"
        )
        log.info(f"🚢 [{thread_name}] Scenario generation completed in {time.time() - gen_start:.2f}s!")
    except Exception as e:
        log.error(f"❌ [{thread_name}] Scenario generation failed for {formatted_zst_name}: {e}")
        if os.path.exists(formatted_path):
            try:
                os.remove(formatted_path)
            except Exception:
                pass
        return False

    # -----------------------------------------------------------------------
    # Step 5: IMMEDIATELY REMOVE Formatted .csv.zst File from Local Disk
    # -----------------------------------------------------------------------
    log.info(f"🧹 [{thread_name}] Step 5/7: Removing intermediate .csv.zst file ({formatted_bytes / (1024*1024):.2f} MB)...")
    if os.path.exists(formatted_path):
        try:
            os.remove(formatted_path)
            log.info(f"   Deleted formatted file: {os.path.basename(formatted_path)}")
        except Exception as e:
            log.warning(f"   Could not remove {formatted_path}: {e}")

    # -----------------------------------------------------------------------
    # Step 6: Upload Generated CommonOcean XML Scenarios to Google Drive
    # -----------------------------------------------------------------------
    generated_xml_files = glob.glob(os.path.join(thread_scenarios_dir, "*.xml"))
    log.info(f"📤 [{thread_name}] Step 6/7: Uploading {len(generated_xml_files)} generated XML scenarios to Google Drive...")

    scenarios_gdrive_folder_id = extract_gdrive_id(scenarios_gdrive_url)
    uploaded_count = 0

    for xml_file in generated_xml_files:
        xml_name = os.path.basename(xml_file)
        success = upload_processed_to_gdrive(
            local_file_path=xml_file,
            output_folder_id_or_url=scenarios_gdrive_folder_id,
            credentials_path=credentials_path,
            mimetype="application/xml"
        )
        if success:
            uploaded_count += 1
            # Step 7: IMMEDIATELY REMOVE uploaded XML scenario from local disk
            try:
                os.remove(xml_file)
                log.info(f"   Deleted uploaded scenario XML: {xml_name}")
            except Exception as e:
                log.warning(f"   Could not remove uploaded XML {xml_name}: {e}")
        else:
            log.warning(f"💾 Preserving un-uploaded scenario XML locally: {xml_name}")

    log.info(f"📤 [{thread_name}] Uploaded {uploaded_count}/{len(generated_xml_files)} XML scenario files to Google Drive folder '{scenarios_gdrive_folder_id}'.")

    # -----------------------------------------------------------------------
    # Step 7: Clean Thread Staging Directories & Free RAM
    # -----------------------------------------------------------------------
    log.info(f"🧹 [{thread_name}] Step 7/7: Cleaning local thread staging workspace...")
    try:
        shutil.rmtree(thread_raw_dir, ignore_errors=True)
        shutil.rmtree(thread_formatted_dir, ignore_errors=True)
        shutil.rmtree(thread_scenarios_dir, ignore_errors=True)
    except Exception as e:
        log.warning(f"   Notice clearing thread directories: {e}")

    gc.collect()
    log.info(f"✨ [{thread_name}] [{idx}/{total_files}] File processing complete! Thread ready for next file.\n")
    return True


# ---------------------------------------------------------------------------
# Master Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_master_pipeline(
    raw_gdrive_url=DEFAULT_RAW_GDRIVE_URL,
    scenarios_gdrive_url=DEFAULT_SCENARIOS_GDRIVE_URL,
    num_threads=DEFAULT_NUM_THREADS,
    credentials_path=None
):
    """Orchestrates multi-threaded end-to-end AIS conversion and scenario generation."""
    log.info("=========================================================================")
    log.info("🚢 MASTER MARITIME AIS PIPELINE: MULTI-THREADED END-TO-END CONVERTER")
    log.info("=========================================================================")
    log.info(f"Input Raw AIS Google Drive:       {raw_gdrive_url}")
    log.info(f"Output Scenarios Google Drive:   {scenarios_gdrive_url}")
    log.info(f"Parallel Worker Threads:         {num_threads} Threads")
    log.info("Strategy:                        0-Disk-Accumulation Streaming Loop")
    log.info("=========================================================================\n")

    # 1. Ensure Coastline Shapefile
    shp_filepath = ensure_coastline_shapefile()

    # 2. Pre-authenticate Google Drive Service on main thread (pop-up OAuth browser prompt once if needed)
    get_gdrive_api_service(credentials_path)

    # 3. Retrieve raw file metadata from Google Drive input folder without downloading
    file_items = get_gdrive_folder_file_list(raw_gdrive_url, credentials_path)

    if not file_items:
        log.warning(f"⚠️ Could not list remote folder files online. Checking local raw directory '{LOCAL_RAW_DOWNLOADS_DIR}'...")
        local_existing = glob.glob(os.path.join(LOCAL_RAW_DOWNLOADS_DIR, "*"))
        file_items = [{'id': None, 'name': os.path.basename(f), 'local_path': f} for f in local_existing if os.path.isfile(f)]

    if not file_items:
        log.error("❌ No raw AIS files found to process.")
        return

    total_files = len(file_items)
    log.info(f"🔄 Starting multi-threaded pipeline for {total_files} total raw files across {num_threads} worker threads...\n")

    # 4. Multithreaded Execution via ThreadPoolExecutor
    num_threads = max(1, int(num_threads))
    with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="AISWorker") as executor:
        futures = [
            executor.submit(
                process_file_worker,
                item, idx, total_files, shp_filepath, raw_gdrive_url, scenarios_gdrive_url, credentials_path
            )
            for idx, item in enumerate(file_items, 1)
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"❌ Thread execution error: {e}")

    os.system('sync')
    log.info("\n🎉 All raw AIS datasets converted to CommonOcean XML Scenarios, uploaded to Google Drive, and local space freed up!")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Multi-Threaded AIS Preprocessing & CommonOcean Scenario Generator Pipeline.")
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS, help=f"Number of parallel worker threads (default: {DEFAULT_NUM_THREADS})")
    parser.add_argument("--gdrive-input", type=str, default=DEFAULT_RAW_GDRIVE_URL, help="Input Raw AIS Google Drive Folder URL or ID")
    parser.add_argument("--gdrive-scenarios-output", type=str, default=DEFAULT_SCENARIOS_GDRIVE_URL, help="Output Scenarios Google Drive Folder URL or ID")
    parser.add_argument("--credentials", type=str, default=None, help="Path to Google Drive OAuth2 / Service Account credentials JSON")

    args = parser.parse_args()

    run_master_pipeline(
        raw_gdrive_url=args.gdrive_input,
        scenarios_gdrive_url=args.gdrive_scenarios_output,
        num_threads=args.num_threads,
        credentials_path=args.credentials
    )
