import pandas as pd
import numpy as np
import sys

def format_mendeley_dataset(input_csv, output_zst="LA_ais.csv.zst"):
    print(f"Loading {input_csv}...")
    
    # We only load the columns we actually need to save RAM
    cols_to_keep = [
        'MMSI', 'timestamp', 'Latitude', 'Longitude', 
        'speed', 'Cog', 'TrueHeading', 'ShipType', 
        'DimensionA', 'DimensionB', 'DimensionC', 'DimensionD', 
        'MaximumStaticDraught'
    ]
    
    # Load the CSV
    df = pd.read_csv(input_csv, usecols=cols_to_keep)
    
    print("Formatting columns...")
    
    # 1. Rename columns to match the ML pipeline
    df = df.rename(columns={
        'MMSI': 'mmsi',
        'timestamp': 'base_date_time',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'speed': 'sog',
        'Cog': 'cog',
        'TrueHeading': 'heading',
        'ShipType': 'vessel_type',
        'MaximumStaticDraught': 'draft'
    })
    
    # 2. Calculate actual length and width from the GPS offset dimensions
    df['length'] = df['DimensionA'] + df['DimensionB']
    df['width'] = df['DimensionC'] + df['DimensionD']
    
    # If dimensions are 0 (missing in AIS), replace with NaN so our main pipeline imputes them to 150x25m
    df.loc[df['length'] == 0, 'length'] = np.nan
    df.loc[df['width'] == 0, 'width'] = np.nan
    
    # Drop the raw dimensions now that we have length/width
    df = df.drop(columns=['DimensionA', 'DimensionB', 'DimensionC', 'DimensionD'])
    
    print(f"Saving heavily compressed dataset to {output_zst}...")
    # Save as Zstandard compressed CSV (This shrinks a 5GB CSV into ~300MB and loads instantly)
    df.to_csv(output_zst, index=False, compression='zstd')
    print("Done! You can now run this file through asidataconvertornew.py")

if __name__ == "__main__":
    # Change "Mendeley_Singapore.csv" to whatever your downloaded file is named
    format_mendeley_dataset("/run/media/akshat/Akshat_USB/ais-2026-01-01.csv", "/run/media/akshat/Akshat_USB/ais.csv.zst")