# Complete Guide: Downloading Raw AIS Data for Scenario Generation

This guide provides instructions on where and how to download raw **AIS (Automatic Identification System)** vessel tracking data for generating maritime navigation scenarios.

---

## 1. Primary Sources for Free Public AIS Data

### A. NOAA / Marine Cadastre (US Coastal Waters & Gulf of Mexico)
- **Coverage**: US coastal waters, Houston Ship Channel, Gulf of Mexico, Atlantic/Pacific coastlines.
- **Update Frequency**: Daily / Monthly archives.
- **Format**: CSV files containing `mmsi, base_date_time, latitude, longitude, sog, cog, heading, vessel_type, length, width, draft`.
- **Download Page**: [https://marinecadastre.gov/ais/](https://marinecadastre.gov/ais/)

#### Downloading via Command Line:
```bash
# Create directory for raw AIS files
mkdir -p /home/akshat/raw_ais_data && cd /home/akshat/raw_ais_data

# Download sample day (e.g. January 1, 2023 - Gulf of Mexico Zone 15):
curl -O https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip

# Unzip raw CSV
unzip AIS_2023_01_01.zip

# Compress to Zstandard (.csv.zst) format expected by scenario generator:
zstd -10 AIS_2023_01_01.csv -o /home/akshat/ais_data_gulf.csv.zst
```

---

### B. Danish Maritime Authority (European / English Channel & Baltic Sea)
- **Coverage**: High-density European straits, Baltic Sea, North Sea transits.
- **Format**: Daily CSV / NMEA archives.
- **Download Page**: [https://dma.dk/safety-at-sea/navigational-information/ais-data](https://dma.dk/safety-at-sea/navigational-information/ais-data)

```bash
# Example download for Baltic Sea AIS:
curl -O https://dma.dk/ais-data/aisdk-2023-01-01.zip
unzip aisdk-2023-01-01.zip
zstd -10 aisdk-2023-01-01.csv -o /home/akshat/ais_data_baltic.csv.zst
```

---

### C. Google Drive / Private Repositories
If raw AIS files are stored on Google Drive:
```bash
pip install gdown

# Download ZST file directly by Google Drive File ID:
gdown --id "YOUR_GOOGLE_DRIVE_FILE_ID" -O /home/akshat/ais_data_2015_3_11.csv.zst
```

---

## 2. Using Downloaded AIS Data in Scenario Generator

Once you have downloaded and compressed `.csv.zst` files:

1. Edit the input path in `SceneriosGenerator/aisdatageneratoroffline.py`:
   ```python
   input_zst_file = "/home/akshat/ais_data_gulf.csv.zst"
   output_directory = "./generated_scenarios_gulf"
   ```

2. Run scenario generation:
   ```bash
   python SceneriosGenerator/aisdatageneratoroffline.py
   ```

3. Merge all generated scenario directories into the final training dataset:
   ```bash
   python helpers/mergescenarios.py
   ```
