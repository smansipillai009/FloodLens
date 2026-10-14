import sys, numpy as np, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
run = sys.argv[1]
def rd(n):
    with rasterio.open(f"{run}/{n}") as s:
        k = max(1, int(np.ceil(max(s.width, s.height) / 1200)))
        return s.read(1, out_shape=(s.height // k, s.width // k)).astype("float32")
fig, ax = plt.subplots(1, 3, figsize=(15, 7))
ax[0].imshow(rd("sar_change_vv.tif"), cmap="seismic", vmin=-6, vmax=6); ax[0].set_title("VV change (blue = darker)")
ax[1].imshow(rd("unet_water_probability.tif"), cmap="viridis", vmin=0, vmax=1); ax[1].set_title("U-Net water probability")
ax[2].imshow(rd("flood_class.tif"), cmap="nipy_spectral"); ax[2].set_title("final classes")
plt.savefig("compare_unet.png", dpi=90, bbox_inches="tight")