"script to reconstruct the image"
import numpy as np
import torch
import rasterio as rio

# Load torch tensor
Xs = torch.load("tensor_b8a_us.pt")
patch_size_h = Xs.shape[1]
patch_size_w = Xs.shape[2]

# Get size of upsampled image
with rio.open("SENTINEL2X_20181015-000000-000_L3A_T31TCJ_D_V1-1_FRC_B8.tif") as src:
    height, width = src.height, src.width
nb_patch_h = height // Xs.shape[1]
nb_patch_w = width // Xs.shape[2]

# Image initialization
IM = np.zeros((nb_patch_h * patch_size_h, nb_patch_w * patch_size_w), dtype=np.int16)

for i in range(nb_patch_h):
    for j in range(nb_patch_w):
        patch_number = i * nb_patch_w + j
        IM[
            i * patch_size_h : (i + 1) * patch_size_h,
            j * patch_size_w : (j + 1) * patch_size_w,
        ] = Xs[i + j * nb_patch_h].numpy()
print(patch_number)

with rio.open("SENTINEL2X_20181015-000000-000_L3A_T31TCJ_D_V1-1_FRC_B8.tif") as src:
    profile = src.profile
    profile.update(
        {
            "height": int(nb_patch_h * patch_size_h),
            "width": int(nb_patch_w * patch_size_w),
        }
    )
    with rio.open("test_us.tif", "w", **profile) as dst:
        dst.write(IM[np.newaxis, ...])
