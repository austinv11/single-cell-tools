from pathlib import Path

import click
import cv2
import numpy as np
import slideio
import tifffile

# File extension to assumed driver:
driver_mappings = {
    "ome.tiff": "OMETIFF",
    "ome.tif": "OMETIFF",
    "afi": "AFI",
    "svs": "SVS",
    "scn": "SCN",
    "tif": "GDAL",
    "tiff": "GDAL",
    "jpg": "GDAL",
    "jpeg": "GDAL",
    "png": "GDAL",
    "bmp": "GDAL",
    "zvi": "ZVI",
    "dicom": "DCM",
    "": "DCM",  # Assume that no extension is a directory containing DICOM images
    "ndpi": "NDPI",
    "vsi": "VSI",
    "qptiff": "QPTIFF",
    "qptif": "QPTIFF",
}


def get_extension(file_path: Path) -> str:
    return "".join(file_path.suffixes).lstrip(".").lower()


def _default_outfile(infile: Path) -> Path:
    stem = infile.name
    for suffix in infile.suffixes:
        stem = stem.removesuffix(suffix)
    return infile.parent / (stem + ".ome.tiff")


def _half_res(image: np.ndarray, photometric: str) -> np.ndarray:
    """Downsample image by 2x using INTER_AREA."""
    if photometric == "minisblack" and image.ndim >= 3:
        # (C, H, W) → (H, W, C) for cv2, then back
        image = np.moveaxis(image, 0, -1)
        h = int(np.floor(image.shape[0] * 0.5))
        w = int(np.floor(image.shape[1] * 0.5))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
    else:
        h = int(np.floor(image.shape[0] * 0.5))
        w = int(np.floor(image.shape[1] * 0.5))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image


def _write_pyramid(
    outfile: Path,
    image: np.ndarray,
    photometric: str,
    metadata: dict,
    subresolutions: int,
) -> None:
    px_size_x = metadata.get("PhysicalSizeX", 1.0)
    px_size_y = metadata.get("PhysicalSizeY", 1.0)

    options = dict(
        photometric=photometric,
        tile=(1024, 1024),
        maxworkers=4,
        compression="jpeg2000",
        compressionargs={"level": 85},
        resolutionunit="CENTIMETER",
    )

    with tifffile.TiffWriter(outfile, bigtiff=True) as tif:
        click.echo("Writing pyramid level 0")
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / px_size_x, 1e4 / px_size_y),
            metadata=metadata,
            **options,
        )

        scale = 1.0
        current = image
        for i in range(subresolutions):
            scale /= 2
            current = _half_res(current, photometric)
            click.echo(f"Writing pyramid level {i + 1}")
            tif.write(
                current,
                subfiletype=1,
                resolution=(1e4 / (scale * px_size_x), 1e4 / (scale * px_size_y)),
                **options,
            )


def convert_generic(infile: Path, outfile: Path, subresolutions: int = 6) -> None:
    extension = get_extension(infile)
    if extension not in driver_mappings:
        raise ValueError("Invalid file extension, must choose from: " + ", ".join(driver_mappings.keys()))

    driver = driver_mappings[extension]

    with slideio.open_slide(str(infile.absolute()), driver) as slide:
        if slide.num_scenes > 1:
            print(f"Warning: {infile} contains multiple scenes, only the first will be converted.")
        scene = slide.get_scene(0)

        num_z = scene.num_z_slices
        num_t = scene.num_t_frames

        if num_z == 1 and num_t == 1:
            image = scene.read_block()
        elif num_t == 1:
            slices = [scene.read_block(slices=(z, z + 1)) for z in range(num_z)]
            image = np.stack(slices, axis=0)
        else:
            frames = []
            for t in range(num_t):
                z_slices = [scene.read_block(slices=(z, z + 1), frames=(t, t + 1)) for z in range(num_z)]
                frames.append(np.stack(z_slices, axis=0))
            image = np.stack(frames, axis=0)

        # Determine photometric and normalise channel axis position.
        # RGB: slideio returns (H, W, 3) uint8 → keep as-is, photometric='rgb'
        # Multichannel: (H, W, C) → move to (C, H, W), photometric='minisblack'
        # Grayscale: (H, W) → photometric='minisblack'
        if image.ndim >= 3 and image.shape[-1] == 3 and scene.num_channels == 3:
            photometric = "rgb"
        elif image.ndim >= 3:
            photometric = "minisblack"
            # move channel axis from last to first spatial position: (..., H, W, C) → (..., C, H, W)
            image = np.moveaxis(image, -1, -3)
        else:
            photometric = "minisblack"

        metadata: dict = {}

        channel_names = [scene.get_channel_name(i) for i in range(scene.num_channels)]
        if any(channel_names):
            metadata["Channel"] = {"Name": channel_names}

        res_x, res_y = scene.resolution
        if res_x > 0 and res_y > 0:
            # slideio resolution is in meters/pixel; OME uses µm
            metadata["PhysicalSizeX"] = res_x * 1e6
            metadata["PhysicalSizeXUnit"] = "µm"
            metadata["PhysicalSizeY"] = res_y * 1e6
            metadata["PhysicalSizeYUnit"] = "µm"

        z_res = scene.z_resolution
        if z_res > 0:
            metadata["PhysicalSizeZ"] = z_res * 1e6
            metadata["PhysicalSizeZUnit"] = "µm"

        _write_pyramid(outfile, image, photometric, metadata, subresolutions)


@click.command()
@click.argument("infile", type=click.Path(exists=True, path_type=Path))
@click.argument("outfile", type=click.Path(path_type=Path), required=False, default=None)
@click.option("--subresolutions", default=7, show_default=True, help="Number of pyramid downsampling levels to write.")
def main(infile: Path, outfile: Path | None, subresolutions: int) -> None:
    """Convert a microscopy image to a Xenium-compatible OME-TIFF pyramid.

    INFILE may be any format supported by slideio (SVS, NDPI, VSI, SCN, ZVI,
    OME-TIFF, standard TIFF/PNG/JPG, DICOM, …).  OUTFILE defaults to the same
    directory and stem as INFILE with a .ome.tiff extension.
    """
    resolved = outfile or _default_outfile(infile)
    convert_generic(infile, resolved, subresolutions=subresolutions)
    click.echo(f"Written: {resolved}")


if __name__ == "__main__":
    main()