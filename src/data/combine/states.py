import os
import glob
import re
import numpy as np
import rasterio
from tqdm import tqdm

from src.config_utils import load_config, load_data_config
from src.processing import regrid

# ============================================================
# 1. BUI STREAMER (Multi-Band + Optional Interpolation)
# ============================================================
class BuiStreamer:
    def __init__(self, bui_dir, start_year, end_year, alpha, interpolate=False,
                 res_km=4):
        self.bui_dir = bui_dir
        self.years = range(start_year, end_year + 1)
        self.alpha = alpha
        self.state = None
        self.interpolate = interpolate

        # Index the per-year multiband quantile GeoTIFFs written by
        # src/data/preprocess/bui.py, e.g. "2020_BUI_4km.tif" (7 linear-space
        # quantile bands). Replaces the old per-band viridis PNGs, which were
        # read via src.read(1) — i.e. the RED CHANNEL of a colormap of
        # x^0.25(quantile). These are the real quantile values; any power
        # transform is deferred to model-input time downstream.
        self.anchors = {}
        print(f"[BUI] Scanning for {res_km}km quantile GeoTIFFs in {bui_dir}...")
        pattern = re.compile(rf".*?(\d{{4}})_BUI_{res_km}km\.tif$")
        for f in glob.glob(os.path.join(bui_dir, f"*_BUI_{res_km}km.tif")):
            m = pattern.match(os.path.basename(f))
            if m:
                self.anchors[int(m.group(1))] = f

        self.sorted_years = sorted(self.anchors.keys())
        print(f"[BUI] Found {len(self.sorted_years)} anchor years.")

    def _load_year_stack(self, year):
        """Load the 7-band quantile GeoTIFF for a year. Returns (H, W, 7) or None."""
        fpath = self.anchors.get(year)
        if fpath is None:
            return None
        try:
            with rasterio.open(fpath) as src:
                arr = src.read().astype(np.float32)  # (bands, H, W)
            return np.transpose(arr, (1, 2, 0))       # -> (H, W, bands)
        except Exception as e:
            print(f"Error loading BUI {year}: {e}")
            return None

    def _get_data_for_year(self, year):
        """Handles interpolation or raw loading."""
        # 1. Try Direct Load
        raw = self._load_year_stack(year)
        if raw is not None:
            return raw
            
        # 2. If Interpolation is OFF, return None (Persistence)
        if not self.interpolate:
            return None
            
        # 3. Interpolation Logic
        past = [y for y in self.sorted_years if y < year]
        future = [y for y in self.sorted_years if y > year]
        
        if not past and not future: return None
        if not past: return self._load_year_stack(future[0])
        if not future: return self._load_year_stack(past[-1])
        
        y1, y2 = past[-1], future[0]
        data1 = self._load_year_stack(y1)
        data2 = self._load_year_stack(y2)
        
        if data1 is None or data2 is None: return None
        
        w = (year - y1) / (y2 - y1)
        return (1 - w) * data1 + w * data2

    def __iter__(self):
        print(f"[BUI] Stream started ({self.years.start}-{self.years.stop-1})...")
        for year in self.years:
            curr = self._get_data_for_year(year)
            
            # Update EMA State
            if self.state is None:
                if curr is not None:
                    self.state = curr
            else:
                if curr is not None:
                    self.state = self.alpha * curr + (1 - self.alpha) * self.state
                
            yield year, self.state

# ============================================================
# 2. PRISM STREAMER (Strict Variables + Bio-Year)
# ============================================================
class PrismStreamer:
    def __init__(self, prism_dir, start_year, end_year, alpha, target_res_m=None):
        self.prism_dir = prism_dir
        self.years = range(start_year, end_year + 1)
        self.alpha = alpha
        self.state = None

        self.VARS = ['ppt', 'tdmean', 'tmax', 'tmean', 'tmin', 'vpdmax', 'vpdmin']
        self.file_template = "prism_{var}_us_25m_{date}_bui4km.tif"

        # Aggregate the 4 km PRISM rasters to the model grid on load (linear
        # mean -- climate aggregates linearly; the encoder standardizes later),
        # mirroring the eBird aggregation in ESK so PRISM and BUI states share
        # the 16 km grid. block is derived on first read from the raster's res.
        self.target_res_m = target_res_m
        self.block = None

    def _load_month_stack(self, yyyymm):
        """Loads all 7 variables for a specific month, aggregated to the grid."""
        stack = []
        for var in self.VARS:
            fname = self.file_template.format(var=var, date=yyyymm)
            fpath = os.path.join(self.prism_dir, fname)

            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing required PRISM file: {fname}")

            with rasterio.open(fpath) as src:
                band = src.read(1).astype(np.float64)
                if self.block is None:
                    native = abs(src.transform.a)
                    self.block = (regrid.block_factor(native, self.target_res_m)
                                  if self.target_res_m else 1)
            if self.block > 1:
                band = regrid.block_reduce(band, self.block, how="mean")
            stack.append(band.astype(np.float32))

        return np.stack(stack, axis=-1)

    def _get_bio_year_stack(self, target_year):
        """Constructs stack for Aug(T-1) -> Jul(T)."""
        prev_year = target_year - 1
        
        # Months: Aug-Dec (Prev) + Jan-Jul (Curr)
        month_keys = []
        for m in ["08", "09", "10", "11", "12"]:
            month_keys.append(f"{prev_year}{m}")
        for m in ["01", "02", "03", "04", "05", "06", "07"]:
            month_keys.append(f"{target_year}{m}")
            
        full_year_stack = []
        try:
            for date_key in month_keys:
                m_stack = self._load_month_stack(date_key)
                full_year_stack.append(m_stack)
        except FileNotFoundError:
            return None
            
        return np.concatenate(full_year_stack, axis=-1)

    def __iter__(self):
        print(f"[PRISM] Stream started ({self.years.start}-{self.years.stop-1})...")
        print(f"[PRISM] Variables: {self.VARS} (Total 84 channels/year)")
        
        for year in self.years:
            curr = self._get_bio_year_stack(year)
            
            if self.state is None:
                if curr is not None:
                    self.state = curr
            else:
                if curr is not None:
                    self.state = self.alpha * curr + (1 - self.alpha) * self.state
                
            yield year, self.state

# ============================================================
# 3. SYNCHRONIZED EXECUTION
# ============================================================
def run_simulation(prism_dir, bui_dir, out_dir, interpolate_bui=False, res_km=None):
    os.makedirs(out_dir, exist_ok=True)
    states_out_dir = os.path.join(out_dir, "yearly_states")
    os.makedirs(states_out_dir, exist_ok=True)
    
    # Configuration
    START_YEAR = 1896  # Needs Aug 1895
    END_YEAR = 2024
    SAMPLE_START = 1900
    EMA_TAU = 10.0
    ALPHA = 1.0 - np.exp(-1.0 / EMA_TAU)
    SAMPLES_PER_YEAR = 20000
    
    if res_km is None:
        res_km = load_data_config()["grid"]["target_res_m"] // 1000  # single source of truth
    target_res_m = res_km * 1000

    # 1. Setup Mask (at the model grid, matching the aggregated PRISM state)
    ref_files = glob.glob(os.path.join(prism_dir, "prism_ppt_*.tif"))
    if not ref_files:
        raise FileNotFoundError("Could not find any PRISM files for mask generation.")
    ref_path = ref_files[0]

    print(f"Using reference file for mask: {os.path.basename(ref_path)}")
    with rasterio.open(ref_path) as src:
        ref_data = src.read(1).astype(np.float64)
        block = regrid.block_factor(abs(src.transform.a), target_res_m)
    if block > 1:  # aggregate to the model grid so sample indices match the state
        ref_data = regrid.block_reduce(ref_data, block, how="mean")
    valid_y, valid_x = np.where(ref_data > -1000)
    valid_coords = list(zip(valid_y, valid_x))

    print(f"Valid Land Pixels (at {res_km} km): {len(valid_coords)}")

    # 2. Initialize Generators (both aggregate their inputs to the model grid)
    gen_prism = PrismStreamer(prism_dir, START_YEAR, END_YEAR, ALPHA,
                              target_res_m=target_res_m)
    gen_bui = BuiStreamer(bui_dir, START_YEAR, END_YEAR, ALPHA,
                          interpolate=interpolate_bui, res_km=res_km)
    
    all_vectors = []
    
    # 3. Lockstep Iteration
    print(f"Starting Simulation. Saving yearly states to {states_out_dir}...")
    
    for (y_p, s_p), (y_b, s_b) in zip(gen_prism, gen_bui):
        assert y_p == y_b, f"Sync Error: {y_p} != {y_b}"
        year = y_p
        
        # Proceed only if both states are initialized
        if s_p is not None and s_b is not None:
            
            if year >= SAMPLE_START:
                # --- A. Random Sampling (for Autoencoder Training) ---
                indices = np.random.choice(len(valid_coords), SAMPLES_PER_YEAR, replace=False)
                rows = [valid_coords[i][0] for i in indices]
                cols = [valid_coords[i][1] for i in indices]
                
                p_vec = s_p[rows, cols] # (N, 84)
                b_vec = s_b[rows, cols] # (N, 7)
                
                # Strict NaN Filter
                mask = ~np.isnan(p_vec).any(axis=1) & ~np.isnan(b_vec).any(axis=1)
                
                if mask.sum() > 0:
                    combined = np.concatenate([p_vec[mask], b_vec[mask]], axis=1)
                    all_vectors.append(combined)

                # --- B. Save Whole Grid State (for Inference Cube) ---
                # We save every valid year to build the full spacetime cube later
                # Format: state_YYYY_bio_ema10.npz
                state_fname = f"state_{year}_bio_ema10.npz"
                np.savez_compressed(
                    os.path.join(states_out_dir, state_fname),
                    prism=s_p, 
                    bui=s_b
                )
        
        if year % 5 == 0:
            print(f"Processed {year}...")

    # 4. Save Final Training Bag
    if all_vectors:
        full_arr = np.concatenate(all_vectors, axis=0)
        print(f"Saving {len(full_arr)} historical training vectors to {out_dir}...")
        np.save(os.path.join(out_dir, "history_vectors_bio_ema10.npy"), full_arr.astype(np.float32))
    else:
        print("WARNING: No vectors sampled! Check data availability.")

def main():
    DATA_DIR = load_config()["paths"]["data_dir"]
    run_simulation(
        prism_dir=f"{DATA_DIR}/prism_monthly_4km_albers",
        # New: the per-year 7-band quantile GeoTIFFs from preprocess/bui.py
        # (was the old BUI_4km_interp viridis PNG directory).
        bui_dir=f"{DATA_DIR}/HBUI",
        out_dir=f"{DATA_DIR}/smoothed_prism_bui",
        interpolate_bui=False,
    )


if __name__ == "__main__":
    main()