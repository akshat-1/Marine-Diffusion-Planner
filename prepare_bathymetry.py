import numpy as np
import netCDF4 as nc
import os

def convert_gebco_to_npz(nc_filepath, npz_filepath):
    if not os.path.exists(nc_filepath):
        print(f"Error: Could not find {nc_filepath}")
        return

    print(f"Loading raw GEBCO data from {nc_filepath}...")
    dataset = nc.Dataset(nc_filepath)

    # GEBCO standard variable names
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    elevation = dataset.variables['elevation'][:]

    # Calculate grid start points and step sizes
    lat0 = float(lats[0])
    lon0 = float(lons[0])
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])

    print("Converting elevations to positive depths and masking land...")
    
    # Convert elevation to depth (GEBCO: negative is underwater, positive is land)
    depth = -elevation.astype(np.float32)
    
    # Anything that is land (elevation > 0) becomes NaN to speed up our pipeline
    depth[depth <= 0] = np.nan

    print(f"Compressing and saving to {npz_filepath}...")
    np.savez_compressed(
        npz_filepath,
        depth=depth.data, # .data strips the NetCDF mask array wrapper
        lat0=lat0,
        lon0=lon0,
        dlat=dlat,
        dlon=dlon
    )
    
    print("Done!")
    print("-" * 30)
    print(f"Grid shape:      {depth.shape}")
    print(f"Latitude range:  {lats[0]:.4f} to {lats[-1]:.4f}")
    print(f"Longitude range: {lons[0]:.4f} to {lons[-1]:.4f}")
    print(f"File size:       {os.path.getsize(npz_filepath) / (1024 * 1024):.1f} MB")

if __name__ == "__main__":
    # UPDATE THIS to match the file you extracted from GEBCO
    INPUT_GEBCO_FILE = "GEBCO_16_Aug_2026_0358c7481d60/gebco_2026_n1.5_s1.1_w103.5_e104.2.nc" 
    
    OUTPUT_NPZ_FILE = "bathymetry/Singaporebathymetry.npz"
    
    convert_gebco_to_npz(INPUT_GEBCO_FILE, OUTPUT_NPZ_FILE)