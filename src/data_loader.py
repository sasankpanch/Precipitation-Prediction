import os
import re
import time
import logging
import h5py
import xarray as xr
import pandas as pd
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
RAW_PATH = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
OUT_PATH = Path(os.getenv("DATA_PROCESSED_DIR", "./data/processed"))
OUT_PATH.mkdir(parents=True, exist_ok=True)

#                                                                                           <- Crop coordinates for RGV
RGV_LAT = slice(25.61, 26.63)
RGV_LON = slice(-98.78, -96.77)

# T                                                                                         <- trying as many worker to split the load for speed
DASK_WORKERS = int(os.getenv("DASK_WORKERS", 8))


def detect_imerg_version(filepath: Path) -> tuple[str | None, str]:
    """
    Inspect HDF5 structure to determine IMERG version and group path.
    - V06B: data lives under 'Grid'
    - V07B: data lives at root level (no 'Grid' group)
    Returns (group_or_None, version_label)
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if 'Grid' in f:
                return 'Grid', 'V06B'
            else:
                return None, 'V07B'
    except Exception as e:
        logger.warning(f"Could not inspect {filepath.name}: {e}. Assuming V07B.")
        return None, 'V07B (assumed)'


def open_imerg_file(filepath: Path, group: str | None) -> xr.Dataset:
    """
    Open an IMERG .nc4 file with h5netcdf + dask chunks.
    Handles both V06B (group='Grid') and V07B (group=None).
    phony_dims='sort' tolerates non-standard HDF5 dimension layouts in V07B.
    """
    open_kwargs = dict(
        engine="h5netcdf",
        phony_dims="sort",              #                                           <- for V07B flat layout
        chunks={"lat": 50, "lon": 50},  #                                           <- dask chunks
        mask_and_scale=True,
    )

    if group is not None:
        open_kwargs["group"] = group
    else:
        open_kwargs["drop_variables"] = ["time_bnds"]

    return xr.open_dataset(filepath, **open_kwargs)


class RGVDataProcessor:
    def __init__(self):
        self.target_vars = ['precipitationCal', 'precipitation', 'HQprecipitation']

    def parse_date_from_filename(self, filename: str) -> pd.Timestamp | None:
        match = re.search(r'\.(\d{8})-S', filename)
        if match:
            return pd.to_datetime(match.group(1), format='%Y%m%d')
        return None

    @delayed
    def process_single_file(self, filename: str, index: int, total: int) -> str:
        """
        Process one IMERG file (V06B or V07B) -> parquet.
        Decorated with @delayed so dask can schedule it in parallel.
        Returns a status string for logging.
        """
        output_name = filename.replace(".nc4", ".parquet")
        output_path = OUT_PATH / output_name

        if output_path.exists():
            return f"[{index}/{total}] Skipped (exists): {output_name}"

        filepath = RAW_PATH / filename

        try:
            #                                                                                           <- Automatically find file version with group structure
            group, version = detect_imerg_version(filepath)
            ds = open_imerg_file(filepath, group)

            #                                                                                           <- lazy appproach with dask
            ds_rgv = ds.sel(lat=RGV_LAT, lon=RGV_LON)

            #                                                                                           <- Finding precipitation variable
            var_name = next((v for v in self.target_vars if v in ds_rgv.data_vars), None)
            if not var_name:
                available = list(ds_rgv.data_vars)
                return f"[{index}/{total}] WARNING - No precip var in {filename}. Available: {available}"

            #                                                                                           <-  .compute() triggers dask execution for the file's graph
            df = ds_rgv[var_name].squeeze().compute().to_dataframe(name=var_name).reset_index()
            df = df.dropna(subset=[var_name])

            #                                                                                           <- Using date from file name rather than looking through dataset (faster and is more reliable )
            date = self.parse_date_from_filename(filename)
            if date is None:
                return f"[{index}/{total}] WARNING - Could not parse date from {filename}. Skipping."

            df['time'] = date
            df['year'] = date.year
            df['month'] = date.month
            df['day_of_month'] = date.day
            df['day_of_year'] = date.day_of_year
            df['week_of_year'] = date.isocalendar().week
            df['imerg_version'] = version  #                              <- if theres a problem with file version (v6 or v7)

            #                                                             <- Drop fill values (NASA is using -9999.0 for missing)
            df = df[df[var_name] >= 0]

            df.to_parquet(
                output_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

            return f"[{index}/{total}] OK ({version}): {output_name}"

        except Exception as e:
            return f"[{index}/{total}] ERROR on {filename}: {str(e)}"

    def process_all_files(self):
        files = sorted([f for f in os.listdir(RAW_PATH) if f.endswith('.nc4')])
        total_files = len(files)

        if total_files == 0:
            logger.error(f"No .nc4 files found in {RAW_PATH}. Check your .env paths.")
            return

        logger.info(
            f"Starting RGV daily processing for {total_files} files "
            f"using {DASK_WORKERS} workers."
        )
        start = time.time()

        #                                                                               <-  dask task graph  one delayed task per file
        tasks = [
            self.process_single_file(filename, i + 1, total_files)
            for i, filename in enumerate(files)
        ]

        #                                                                               <- Execute all tasks in parallel across cpu cores
        with ProgressBar():
            results = compute(*tasks, scheduler='threads', num_workers=DASK_WORKERS)

        #                                                                                <- Log all results after batch completes (for debugging)
        for msg in results:
            if 'ERROR' in msg:
                logger.error(msg)
            elif 'WARNING' in msg:
                logger.warning(msg)
            else:
                logger.info(msg)

        duration = time.time() - start
        logger.info(f"Full batch complete in {duration:.2f}s across {total_files} files.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', help='Wipe processed dir before running')
    parser.add_argument('--workers', type=int, default=DASK_WORKERS,
                        help=f'Number of dask workers (default: {DASK_WORKERS})')
    args = parser.parse_args()

    if args.workers:
        DASK_WORKERS = args.workers

    if args.clean:
        import shutil
        if OUT_PATH.exists():
            shutil.rmtree(OUT_PATH)
            OUT_PATH.mkdir(parents=True, exist_ok=True)
            print(f"Cleared {OUT_PATH}")

    print("Initializing environment and checking paths...")
    if not RAW_PATH.exists():
        print(f"Directory not found: {RAW_PATH}. Please check your .env file.")
    else:
        processor = RGVDataProcessor()
        processor.process_all_files()