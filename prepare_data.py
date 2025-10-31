"""Script to prepare data for labwork.

We use one sentinel-2 image, at two different wavelenghts but overlapin the same.

The images is first split into patches. Then the image at 10 m/pixel is downscaled to 20
for the learning step: learn how to recover detail from 20m.

The original band at 20 m is then upsampled to 10 m for inference.

More info on Sentinel 2 : https://en.wikipedia.org/wiki/Sentinel-2

"""
from pathlib import Path
import torch
import rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling

current_path = Path(
    "/home/onyxia/work/SuperResolution"
)

output_path = Path(
    "/home/onyxia/work/SuperResolution/Patchs"
)

SIZES = [(64, 64), (32, 32)]

IMAGES = [
    "SENTINEL2X_20181015-000000-000_L3A_T31TCJ_D_V1-1_FRC_B8.tif",
    "SENTINEL2X_20181015-000000-000_L3A_T31TCJ_D_V1-1_FRC_B8A.tif",
]

for image, (xsize, ysize) in zip(IMAGES, SIZES):
    with rio.open(image) as src:
        band = image.split(".")[0].split("_")[-1]
        patch = 0
        for i in range(0, src.height - xsize, xsize):
            if i + xsize < src.height:
                for j in range(0, src.width - ysize, ysize):
                    if j + ysize < src.width:
                        # Define the window
                        window = Window(i, j, xsize, ysize)
                        transform = src.window_transform(window)
                        # Create a new cropped raster to write to
                        profile = src.profile
                        profile.update(
                            {"height": xsize, "width": ysize, "transform": transform}
                        )

                        with rio.open(
                            Path(output_path, f"{band}_{patch}.tif"), "w", **profile
                        ) as dst:
                            # Read the data from the window and write it to the output raster
                            dst.write(src.read(window=window))
                    patch += 1
        print(f"Number of patch for {band} = {patch-1}")


# Downsamples B8
scale = 0.5
for patch in range(29241):  # There is 29240 patches
    with rio.open(Path(output_path, f"B8_{patch}.tif")) as src:
        # scale image transform
        transform = src.transform * src.transform.scale(1.0 / scale, 1.0 / scale)
        profile = src.profile
        profile.update(
            {
                "height": int(src.height * scale),
                "width": int(src.width * scale),
                "transform": transform,
            }
        )
        with rio.open(Path(output_path, f"B8_{patch}_ds.tif"), "w", **profile) as dst:
            # resample data to target shape
            dst.write(
                src.read(
                    out_shape=(
                        src.count,
                        int(src.height * scale),
                        int(src.width * scale),
                    ),
                    resampling=Resampling.bilinear,
                )
            )

# Upsamples B8A
scale = 2
for patch in range(29241):
    with rio.open(Path(output_path, f"B8A_{patch}.tif")) as src:
        # scale image transform
        transform = src.transform * src.transform.scale(1.0 / scale, 1.0 / scale)
        profile = src.profile
        profile.update(
            {
                "height": int(src.height * scale),
                "width": int(src.width * scale),
                "transform": transform,
            }
        )
        with rio.open(Path(output_path, f"B8A_{patch}_us.tif"), "w", **profile) as dst:
            # resample data to target shape
            dst.write(
                src.read(
                    out_shape=(
                        src.count,
                        int(src.height * scale),
                        int(src.width * scale),
                    ),
                    resampling=Resampling.bilinear,
                )
            )

# Create Tensor Data Set for band 8
xsize, ysize = 64, 64
X, Xs = (
    torch.zeros((29241, xsize, ysize)),
    torch.zeros((29241, int(xsize / 2), int(ysize / 2))),
)
for patch in range(29241):
    X[patch, :, :] = torch.tensor(rio.open(Path(output_path, f"B8_{patch}.tif")).read())
    Xs[patch, :, :] = torch.tensor(
        rio.open(Path(output_path, f"B8_{patch}_ds.tif")).read()
    )
torch.save(X, Path(output_path, "tensor_b8.pt"))
torch.save(Xs, Path(output_path, "tensor_b8_ds.pt"))

# Create Tensor Data Set for band 8A
X, Xs = (
    torch.zeros((29241, int(xsize / 2), int(ysize / 2))),
    torch.zeros((29241, xsize, ysize)),
)
for patch in range(29241):
    X[patch, :, :] = torch.tensor(
        rio.open(Path(output_path, f"B8A_{patch}.tif")).read()
    )
    Xs[patch, :, :] = torch.tensor(
        rio.open(Path(output_path, f"B8A_{patch}_us.tif")).read()
    )
torch.save(X, Path(output_path, "tensor_b8a.pt"))
torch.save(Xs, Path(output_path, "tensor_b8a_us.pt"))
