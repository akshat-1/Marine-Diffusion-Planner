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
import subprocess
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

RUN_SAVED_DIR = os.path.join(SCRIPT_DIR, "run_saved")
LOCAL_TEMP_DIR = os.path.join(RUN_SAVED_DIR, "Local_Temp_Data")
LOCAL_RAW_DOWNLOADS_DIR = os.path.join(RUN_SAVED_DIR, "raw_ais_downloads")
LOCAL_FORMATTED_DIR = os.path.join(RUN_SAVED_DIR, "formatted_zst_temp")
LOCAL_SCENARIOS_DIR = os.path.join(RUN_SAVED_DIR, "generated_scenarios_temp")

# Thread-safe global scenario counter to guarantee zero filename conflicts and continuous numbering
GLOBAL_SCENARIO_COUNTER = 0
COUNTER_LOCK = threading.Lock()


def get_next_scenario_index_range(batch_count: int) -> int:
    """Thread-safely acquires a continuous global scenario index range."""
    global GLOBAL_SCENARIO_COUNTER
    with COUNTER_LOCK:
        start_idx = GLOBAL_SCENARIO_COUNTER
        GLOBAL_SCENARIO_COUNTER += batch_count
        return start_idx


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
    credentials_path: str = None,
    node_rank: int = 0,
    world_size: int = 1
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
    # Step 4: Real-Time Streaming Scenario Generation & Google Drive Upload
    # -----------------------------------------------------------------------
    scenario_queue = Queue()
    scenarios_gdrive_folder_id = extract_gdrive_id(scenarios_gdrive_url)
    uploaded_stats = {"count": 0}

    # Asynchronous Real-Time Scenario Uploader Consumer Thread
    def real_time_uploader_worker():
        uploader_thread_name = f"{thread_name}-Uploader"
        log.info(f"🌐 [{uploader_thread_name}] Real-time streaming uploader thread active...")

        while True:
            xml_file = scenario_queue.get()
            if xml_file is None:  # Sentinel signal: scenario generation complete
                scenario_queue.task_done()
                break

            if not os.path.exists(xml_file):
                scenario_queue.task_done()
                continue

            # Assign continuous global scenario ID & rename file
            with COUNTER_LOCK:
                global GLOBAL_SCENARIO_COUNTER
                global_idx = GLOBAL_SCENARIO_COUNTER
                GLOBAL_SCENARIO_COUNTER += 1

            if world_size > 1:
                new_scenario_id = f"scenario_node{node_rank}_{global_idx:06d}"
            else:
                new_scenario_id = f"scenario_{global_idx:06d}"
            new_xml_name = f"{new_scenario_id}.xml"
            new_xml_path = os.path.join(thread_scenarios_dir, new_xml_name)

            # Update internal CommonOcean XML scenarioId attribute
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                updated_content = re.sub(r'scenarioId="[^"]+"', f'scenarioId="{new_scenario_id}"', xml_content)
                with open(new_xml_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                if xml_file != new_xml_path and os.path.exists(xml_file):
                    os.remove(xml_file)
            except Exception as e:
                new_xml_path = xml_file

            # Real-time upload to Google Drive
            log.info(f"📤 [{uploader_thread_name}] Real-time uploading generated scenario: {new_xml_name} (Global Index: #{global_idx})...")
            success = upload_processed_to_gdrive(
                local_file_path=new_xml_path,
                output_folder_id_or_url=scenarios_gdrive_folder_id,
                credentials_path=credentials_path,
                mimetype="application/xml"
            )

            # Immediately delete uploaded XML file from local disk to free space
            if success and os.path.exists(new_xml_path):
                try:
                    os.remove(new_xml_path)
                    log.info(f"🧹 [{uploader_thread_name}] Deleted uploaded XML from disk: {new_xml_name} (Disk space clean)")
                except Exception:
                    pass
                uploaded_stats["count"] += 1
            else:
                log.warning(f"💾 Preserving un-uploaded scenario XML locally: {new_xml_name}")

            scenario_queue.task_done()

        log.info(f"✨ [{uploader_thread_name}] Uploader thread finished!")

    # Start uploader consumer thread
    uploader_thread = threading.Thread(target=real_time_uploader_worker, name=f"{thread_name}-Up", daemon=True)
    uploader_thread.start()

    # Callback invoked immediately when each scenario XML is written
    def on_scenario_created_callback(created_xml_path):
        scenario_queue.put(created_xml_path)

    log.info(f"🚢 [{thread_name}] Step 4/7: Generating CommonOcean XML Scenarios from {formatted_zst_name} (Real-time Uploader Active)...")
    try:
        gen_start = time.time()
        stats = generate_dataset_from_ais(
            zst_filepath=formatted_path,
            shp_filepath=shp_filepath,
            output_dir=thread_scenarios_dir,
            sampling_strategy="density_first",
            on_scenario_created=on_scenario_created_callback
        )
        log.info(f"🚢 [{thread_name}] Scenario generation completed in {time.time() - gen_start:.2f}s!")
    except Exception as e:
        log.error(f"❌ [{thread_name}] Scenario generation failed for {formatted_zst_name}: {e}")
    finally:
        # Signal uploader thread to finish & wait for remaining queue items to be uploaded & deleted
        scenario_queue.put(None)
        scenario_queue.join()
        uploader_thread.join()

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

    log.info(f"📤 [{thread_name}] Step 6/7: Uploaded {uploaded_stats['count']} scenario XML files to Google Drive folder '{scenarios_gdrive_folder_id}'.")

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


def dispatch_remote_worker_node(
    node_rank: int,
    world_size: int,
    worker_host: str,
    raw_gdrive_url: str,
    scenarios_gdrive_url: str,
    num_threads: int,
    credentials_path: str = None
) -> bool:
    """Executes run.py on a remote worker CPU node via SSH and streams stdout logs to Master."""
    project_dir = os.path.abspath(SCRIPT_DIR)
    python_bin = sys.executable or "python3"

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", worker_host,
        f"cd '{project_dir}' && '{python_bin}' run.py --node-rank {node_rank} --world-size {world_size} --num-threads {num_threads} --gdrive-input '{raw_gdrive_url}' --gdrive-scenarios-output '{scenarios_gdrive_url}'"
    ]
    if credentials_path:
        ssh_cmd[-1] += f" --credentials '{credentials_path}'"

    log.info(f"🚀 [MASTER DISPATCH] Launching Worker Node Rank {node_rank}/{world_size} on remote host: {worker_host}")
    log.debug(f"[DEBUG] SSH Command: {' '.join(ssh_cmd)}")

    try:
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            line_str = line.strip()
            if line_str:
                log.info(f"[{worker_host}|Rank{node_rank}] {line_str}")
        proc.wait()
        if proc.returncode == 0:
            log.info(f"✅ [MASTER] Worker Node Rank {node_rank}/{world_size} ({worker_host}) finished successfully!")
            return True
        else:
            log.error(f"❌ [MASTER] Worker Node Rank {node_rank}/{world_size} ({worker_host}) exited with error code {proc.returncode}")
            return False
    except Exception as e:
        log.error(f"❌ [MASTER] Failed SSH dispatch to worker node {worker_host}: {e}")
        return False


# ---------------------------------------------------------------------------
# Master Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_master_pipeline(
    raw_gdrive_url=DEFAULT_RAW_GDRIVE_URL,
    scenarios_gdrive_url=DEFAULT_SCENARIOS_GDRIVE_URL,
    num_threads=DEFAULT_NUM_THREADS,
    credentials_path=None,
    node_rank=0,
    world_size=1,
    worker_hosts=None
):
    """Orchestrates multi-threaded end-to-end AIS conversion and scenario generation across CPU cluster nodes."""
    if worker_hosts and node_rank == 0:
        world_size = 1 + len(worker_hosts)

    log.info("=========================================================================")
    log.info("🚢 MASTER MARITIME AIS PIPELINE: CLUSTER MASTER-WORKER CONVERTER")
    log.info("=========================================================================")
    log.info(f"Input Raw AIS Google Drive:       {raw_gdrive_url}")
    log.info(f"Output Scenarios Google Drive:   {scenarios_gdrive_url}")
    log.info(f"Cluster Configuration:           Node {node_rank} of {world_size} Total Cluster Nodes")
    if worker_hosts and node_rank == 0:
        log.info(f"Managed Worker Hosts ({len(worker_hosts)}):   {', '.join(worker_hosts)}")
    log.info(f"Parallel Worker Threads:         {num_threads} Threads per Node")
    log.info("Strategy:                        Zero-Duplication Modulo Partitioning & Real-Time Streaming")
    log.info("=========================================================================\n")

    # 1. Ensure Coastline Shapefile
    shp_filepath = ensure_coastline_shapefile()

    # 2. Pre-authenticate Google Drive Service on main thread (pop-up OAuth browser prompt once if needed)
    get_gdrive_api_service(credentials_path)

    # 3. If Master Node (Rank 0) and worker_hosts are specified: Dispatch remote workers asynchronously
    remote_executor = None
    remote_futures = []
    if node_rank == 0 and worker_hosts:
        log.info(f"👑 [MASTER NODE] Auto-dispatching pipeline across {len(worker_hosts)} remote worker nodes...")
        remote_executor = ThreadPoolExecutor(max_workers=len(worker_hosts), thread_name_prefix="MasterSSH")
        for idx, host in enumerate(worker_hosts, 1):
            remote_rank = idx
            f = remote_executor.submit(
                dispatch_remote_worker_node,
                remote_rank, world_size, host, raw_gdrive_url, scenarios_gdrive_url, num_threads, credentials_path
            )
            remote_futures.append(f)

    # 4. Retrieve raw file metadata from Google Drive input folder without downloading content
    file_items = get_gdrive_folder_file_list(raw_gdrive_url, credentials_path)

    if not file_items:
        log.warning(f"⚠️ Could not list remote folder files online. Checking local raw directory '{LOCAL_RAW_DOWNLOADS_DIR}'...")
        local_existing = glob.glob(os.path.join(LOCAL_RAW_DOWNLOADS_DIR, "*"))
        file_items = [{'id': None, 'name': os.path.basename(f), 'local_path': f} for f in local_existing if os.path.isfile(f)]

    if not file_items:
        log.error("❌ No raw AIS files found to process.")
        return

    total_remote_files = len(file_items)

    # 5. Multi-Node Cluster Disjoint Partitioning (Zero raw file processing duplication)
    if world_size > 1:
        node_file_items = [
            item for idx, item in enumerate(file_items)
            if (idx % world_size) == node_rank
        ]
        log.info(f"🌐 [CLUSTER MODE] Node Rank {node_rank}/{world_size}: Claimed {len(node_file_items)} of {total_remote_files} total raw files via modulo partitioning.")
        file_items = node_file_items

    if not file_items:
        log.info(f"✨ Node Rank {node_rank}/{world_size}: No raw files assigned to this node rank.")
    else:
        total_files = len(file_items)
        log.info(f"🔄 Starting multi-threaded pipeline for {total_files} assigned raw files across {num_threads} worker threads...\n")

        # 6. Multithreaded Execution via ThreadPoolExecutor for local rank tasks
        num_threads = max(1, int(num_threads))
        with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=f"Node{node_rank}Worker") as executor:
            futures = [
                executor.submit(
                    process_file_worker,
                    item, idx, total_files, shp_filepath, raw_gdrive_url, scenarios_gdrive_url, credentials_path, node_rank, world_size
                )
                for idx, item in enumerate(file_items, 1)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"❌ Thread execution error: {e}")

    # 7. Master waits for all remote worker nodes to finish
    if remote_executor is not None:
        log.info("⏳ [MASTER NODE] Waiting for all remote worker nodes to finish processing...")
        for f in as_completed(remote_futures):
            try:
                f.result()
            except Exception as e:
                log.error(f"❌ Remote worker dispatch error: {e}")
        remote_executor.shutdown(wait=True)

    os.system('sync')
    log.info(f"\n🎉 Node Rank {node_rank}/{world_size} completed! All assigned raw datasets converted, uploaded, and disk space cleaned!")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Auto-detect cluster rank and world size from environment variables (PyTorch, SLURM, MPI, or custom)
    env_world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("NNODES", os.environ.get("SLURM_NNODES", "1"))))
    env_node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", os.environ.get("SLURM_NODEID", "0"))))

    parser = argparse.ArgumentParser(description="Master Multi-Node Cluster AIS Preprocessing & CommonOcean Scenario Generator Pipeline.")
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS, help=f"Number of parallel worker threads per node (default: {DEFAULT_NUM_THREADS})")
    parser.add_argument("--node-rank", type=int, default=env_node_rank, help=f"Cluster Node Rank ID (default: auto-detected env NODE_RANK/RANK or {env_node_rank})")
    parser.add_argument("--world-size", type=int, default=env_world_size, help=f"Total Number of Cluster Nodes (default: auto-detected env WORLD_SIZE/NNODES or {env_world_size})")
    parser.add_argument("--workers", type=str, default=None, help="Comma-separated list of remote worker hostnames/IPs for Master to manage (e.g., '192.168.1.101,192.168.1.102')")
    parser.add_argument("--hosts-file", type=str, default=None, help="Path to text file containing list of worker hostnames/IPs (one per line)")
    parser.add_argument("--gdrive-input", type=str, default=DEFAULT_RAW_GDRIVE_URL, help="Input Raw AIS Google Drive Folder URL or ID")
    parser.add_argument("--gdrive-scenarios-output", type=str, default=DEFAULT_SCENARIOS_GDRIVE_URL, help="Output Scenarios Google Drive Folder URL or ID")
    parser.add_argument("--credentials", type=str, default=None, help="Path to Google Drive OAuth2 / Service Account credentials JSON")

    args = parser.parse_args()

    # Parse worker hosts from --workers flag or --hosts-file
    worker_hosts_list = []
    if args.workers:
        worker_hosts_list = [h.strip() for h in args.workers.split(",") if h.strip()]
    elif args.hosts_file and os.path.exists(args.hosts_file):
        with open(args.hosts_file, "r") as f:
            worker_hosts_list = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    run_master_pipeline(
        raw_gdrive_url=args.gdrive_input,
        scenarios_gdrive_url=args.gdrive_scenarios_output,
        num_threads=args.num_threads,
        credentials_path=args.credentials,
        node_rank=args.node_rank,
        world_size=args.world_size,
        worker_hosts=worker_hosts_list
    )
