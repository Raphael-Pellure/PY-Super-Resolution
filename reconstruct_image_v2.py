"""
Reconstruct image from patches tensor
Usage:
python reconstruct_image.py input_tensor.pt output.tif
"""

import sys
import numpy as np
import torch
import rasterio as rio

# -------- RÉFÉRENCE FIXE --------
REFERENCE_TIF = "SENTINEL2X_20181015-000000-000_L3A_T31TCJ_D_V1-1_FRC_B8.tif"


def reconstruct_image(tensor_path, output_tif):
    # -------- 1) Charger le tenseur --------
    Xs = torch.load(tensor_path)   # (N, H, W) ou (N,1,H,W)
    print("Tensor loaded:", Xs.shape)

    # Enlever canal si présent
    if Xs.dim() == 4:
        Xs = Xs.squeeze(1)

    patch_size_h = Xs.shape[1]
    patch_size_w = Xs.shape[2]

    # -------- 2) Lire taille finale depuis TIFF FIXE --------
    with rio.open(REFERENCE_TIF) as src:
        height, width = src.height, src.width
        profile = src.profile

    nb_patch_h = height // patch_size_h
    nb_patch_w = width // patch_size_w

    print("Expected patches:", nb_patch_h * nb_patch_w)
    print("Tensor patches   :", Xs.shape[0])

    # -------- 3) Créer l’image vide --------
    IM = np.zeros(
        (nb_patch_h * patch_size_h, nb_patch_w * patch_size_w),
        dtype=np.int16,
    )
    # -------- 4) Reconstruction mosaïque --------
    for i in range(nb_patch_h):
        for j in range(nb_patch_w):
            patch_number = i * nb_patch_w + j
            IM[
                i * patch_size_h : (i + 1) * patch_size_h,
                j * patch_size_w : (j + 1) * patch_size_w,
            ] = Xs[patch_number].numpy()

    print("Last patch index:", patch_number)

    # -------- 5) Écriture TIFF --------
    profile.update({
        "height": IM.shape[0],
        "width": IM.shape[1],
        "count": 1,
        "dtype": "int16",
    })

    with rio.open(output_tif, "w", **profile) as dst:
        dst.write(IM[np.newaxis, ...])

    print("✅ Image reconstructed and saved to:", output_tif)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nUsage:")
        print("  python reconstruct_image.py input_tensor.pt output.tif\n")
        print("Référence utilisée :", REFERENCE_TIF)
        sys.exit(1)

    tensor_path = sys.argv[1]
    output_path = sys.argv[2]

    reconstruct_image(tensor_path, output_path)
