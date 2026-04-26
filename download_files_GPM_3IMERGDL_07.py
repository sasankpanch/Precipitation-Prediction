#!/usr/bin/env python3
"""
Download IMERG Daily Late Run V07 data using earthaccess
Covers 2000-06-01 to 2024-12-31, cropped to RGV bbox

Requirements:
    > pip install earthaccess tqdm

To run: `python download_files_GPM_3IMERGDL_07.py`
"""

import earthaccess
from pathlib import Path

DOWNLOAD_DIR = Path("/Volumes/MyPassport/Projects/Precipitation-Daily-Dataset/Raw")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Authenticate — prompts for username/password on first run
    # then saves credentials to ~/.netrc automatically
    print("Authenticating with NASA Earthdata...")
    earthaccess.login(strategy="interactive", persist=True)

    print("Searching for IMERG Daily granules...")
    results = earthaccess.search_data(
        short_name="GPM_3IMERGDL",
        version="07",
        temporal=("2000-06-01", "2024-12-31"),
        bounding_box=(-98.78, 25.61, -96.77, 26.63),
    )

    print(f"Found {len(results):,} files to download")

    print(f"Downloading to {DOWNLOAD_DIR}...")
    earthaccess.download(results, local_path=str(DOWNLOAD_DIR))

    print("✅ All downloads complete.")

if __name__ == "__main__":
    main()