import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import parse_config
import geopandas as gpd
import rasterio
from rasterio.transform import Affine

config = parse_config()
us_map_path = config['data']['us_map_path']
raw_cdl_path = config['data']['raw_cdl_path']
downscale_factor = config['data']['downscale_factor']

def inspect_downscaled_cdl(path: str):
    file_path = Path(path)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    # Load original .tif to get original transform and CRS
    with rasterio.open(raw_cdl_path) as src:
        orig_transform = src.transform
        crs = src.crs

    arr = np.load(file_path)
    us_map = gpd.read_file(us_map_path)
    us_map = us_map.to_crs(crs)

    new_transform = Affine(
        orig_transform.a * downscale_factor,  # pixel width
        orig_transform.b,
        orig_transform.c,
        orig_transform.d,
        orig_transform.e * downscale_factor,  # pixel height
        orig_transform.f
    )

    # Calculate extent for imshow
    extent = (
        new_transform.c,
        new_transform.c + arr.shape[1] * new_transform.a,
        new_transform.f + arr.shape[0] * new_transform.e,
        new_transform.f
    )


    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot a small preview of the top-left corner
    # im = ax.imshow(arr[:preview_size, :preview_size], cmap='Greens', interpolation='nearest')
    im = ax.imshow(arr, cmap='Greens', extent=extent, origin='upper')
    us_map.boundary.plot(ax=ax, color='black', linewidth=1)

    ax.set_title(f"downscaled crop land layout preview")
    plt.colorbar(im, ax=ax, label='Crop density', shrink=0.5)
    ax.set_xlim([-2400000, 2300000])
    ax.set_ylim([250000, 3200000])
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# example usage
if __name__ == "__main__":
    inspect_downscaled_cdl("dataset/processed/cdl/WHEAT_WINTER.npy")