import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import parse_config

def inspect_downscaled_cdl(path: str, preview_size: int = 10000):
    file_path = Path(path)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    arr = np.load(file_path)

    # Plot a small preview of the top-left corner
    plt.imshow(arr[:preview_size, :preview_size], cmap='Greens', interpolation='nearest')
    plt.title(f"Downscaled CDL (Top-Left {preview_size}×{preview_size})")
    plt.colorbar(label='Value')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# example usage
if __name__ == "__main__":
    inspect_downscaled_cdl("../dataset/processed/cdl/CORN.npy")