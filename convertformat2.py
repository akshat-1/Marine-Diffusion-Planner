import os
import pandas as pd
import numpy as np
import io

def clean_csv_lines(input_path, encoding='ISO-8859-1'):
    """
    Generator that yields cleaned CSV lines.
    Skips binary garbage lines and fixes common CSV syntax errors.
    """
    import re
    
    def is_binary_garbage(line, threshold=0.3):
        """Detect if line contains mostly non-printable/binary data."""
        if not line:
            return True
        # Count printable ASCII characters (32-126) + common extended
        printable = sum(1 for c in line if 32 <= ord(c) <= 126 or c in '\t\n\r')
        return (printable / len(line)) < (1 - threshold)
    
    def looks_like_valid_csv(line, expected_fields=17):
        """Quick heuristic: does this look like a valid CSV row?"""
        # Must have roughly the right number of commas
        comma_count = line.count(',')
        if comma_count < expected_fields - 2 or comma_count > expected_fields + 5:
            return False
        # Should not be mostly binary
        if is_binary_garbage(line):
            return False
        return True
    
    with open(input_path, 'r', encoding=encoding, errors='replace') as f:
        header = next(f)
        yield header  # Yield header unchanged
        
        for line_num, line in enumerate(f, start=2):
            original = line.rstrip('\n\r')
            if not original:
                continue
            
            # AGGRESSIVE: Skip binary garbage lines immediately
            if is_binary_garbage(original):
                continue
                
            # Skip lines that don't look like valid CSV structure
            if not looks_like_valid_csv(original):
                continue
            
            # Fix 1: Unbalanced quotes (odd number) - add closing quote at end
            quote_count = original.count('"')
            if quote_count % 2 == 1:
                if not original.endswith('"'):
                    original = original + '"'
            
            # Fix 2: Missing comma after quoted field at end of line
            # Pattern: "value"value -> "value",value
            original = re.sub(r'"([^",\n\r]+)$', r'"\1"', original)
            
            yield original

def format_noaa_dataset(input_csv, output_zst="AISFiles/canal_ais.csv.zst"):
    print(f"Loading and cleaning {input_csv}...")
    
    cols_to_keep = [
        'MMSI', 'BaseDateTime', 'LAT', 'LON', 
        'SOG', 'COG', 'Heading', 'VesselType', 
        'Length', 'Width', 'Draft'
    ]
    
    # 1. Pre-clean CSV lines, then parse with pandas
    os.makedirs(os.path.dirname(output_zst), exist_ok=True)
    
    # Use cleaned lines generator
    cleaned_lines = clean_csv_lines(input_csv)
    
    # Parse from the generator using StringIO
    df = pd.read_csv(
        io.StringIO('\n'.join(cleaned_lines)),
        usecols=cols_to_keep,
        dtype=str,
        engine='python',
        on_bad_lines='warn'
    )
    
    print(f"Loaded {len(df)} rows.")
    
    print("Formatting and cleaning columns...")
    
    # 2. Rename columns
    df = df.rename(columns={
        'MMSI': 'mmsi',
        'BaseDateTime': 'base_date_time',
        'LAT': 'latitude',
        'LON': 'longitude',
        'SOG': 'sog',
        'COG': 'cog',
        'Heading': 'heading',
        'VesselType': 'vessel_type',
        'Length': 'length',
        'Width': 'width',
        'Draft': 'draft'
    })
    
    # --- FIX 1: SANITIZE DATES ---
    # This coerces corrupted dates (like "2022-01286") into NaT (Not a Time) quietly
    df['base_date_time'] = pd.to_datetime(df['base_date_time'], errors='coerce')
    
    # --- FIX 2: SANITIZE MMSI & VESSEL TYPE ---
    # This forces them into numbers, turning any random text into NaN
    for col in ['mmsi', 'vessel_type']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Safely coerce all other numeric columns
    numeric_cols = ['latitude', 'longitude', 'sog', 'cog', 'heading', 'length', 'width', 'draft']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- FIX 3: DELETE CORRUPTED ROWS ---
    # Drop any row where the date, coordinates, or MMSI broke during conversion
    df = df.dropna(subset=['latitude', 'longitude', 'base_date_time', 'mmsi'])
    
    # Now that NaNs are gone, force MMSI and VesselType to strict integers.
    # This prevents the "DtypeWarning" when your old script reads the file.
    df['mmsi'] = df['mmsi'].astype(np.int64)
    df['vessel_type'] = df['vessel_type'].fillna(-1).astype(int)
    
    # Clean up missing vessel dimensions
    df.loc[df['length'] <= 0, 'length'] = np.nan
    df.loc[df['width'] <= 0, 'width'] = np.nan
    
    print(f"Saving heavily compressed dataset to {output_zst}...")
    # Saving standardizes the date format perfectly for the downstream script
    df.to_csv(output_zst, index=False, compression='zstd')
    print("Done! File is perfectly clean. You can now run oldaisdataconverter.py")

if __name__ == "__main__":
    local_output = "AISFiles/canal_ais.csv.zst"
    format_noaa_dataset(
        "/run/media/akshat/Akshat_USB/AISData/data/Los Angeles_anonymized.csv", 
        local_output
    )
    # Ensure flush to disk
    os.system('sync')
    
    # Check integrity
    print(f"Verifying integrity of {local_output}...")
    ret = os.system(f"zstd -t {local_output}")
    if ret == 0:
        print("Integrity check passed!")
        usb_output = "/run/media/akshat/Akshat_USB/AISFiles/canal_ais.csv.zst"
        print(f"Copying to USB: {usb_output}...")
        import shutil
        os.makedirs(os.path.dirname(usb_output), exist_ok=True)
        shutil.copy2(local_output, usb_output)
        os.system('sync')
        print("Done! USB file updated and synced.")
    else:
        print("CRITICAL: Local file is also corrupted! Check memory/disk.")
