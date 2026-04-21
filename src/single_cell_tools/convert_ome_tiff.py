from pathlib import Path

import click
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


def convert_generic(infile: Path, outfile: Path) -> None:
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
            axes = "YXS" if image.ndim == 3 else "YX"
        elif num_t == 1:
            slices = [scene.read_block(slices=(z, z + 1)) for z in range(num_z)]
            image = np.stack(slices, axis=0)
            axes = "ZYXS" if image.ndim == 4 else "ZYX"
        else:
            frames = []
            for t in range(num_t):
                z_slices = [scene.read_block(slices=(z, z + 1), frames=(t, t + 1)) for z in range(num_z)]
                frames.append(np.stack(z_slices, axis=0))
            image = np.stack(frames, axis=0)
            axes = "TZYXS" if image.ndim == 5 else "TZYX"

        metadata: dict = {"axes": axes}

        channel_names = [scene.get_channel_name(i) for i in range(scene.num_channels)]
        if any(channel_names):
            metadata["Channel"] = {"Name": channel_names}

        res_x, res_y = scene.resolution
        if res_x > 0 and res_y > 0:
            # slideio reports resolution in meters/pixel; OME uses microns
            metadata["PhysicalSizeX"] = res_x * 1e6
            metadata["PhysicalSizeXUnit"] = "µm"
            metadata["PhysicalSizeY"] = res_y * 1e6
            metadata["PhysicalSizeYUnit"] = "µm"

        z_res = scene.z_resolution
        if z_res > 0:
            metadata["PhysicalSizeZ"] = z_res * 1e6
            metadata["PhysicalSizeZUnit"] = "µm"

        tifffile.imwrite(
            outfile,
            image,
            bigtiff=True,
            metadata=metadata,
        )


@click.command()
@click.argument("infile", type=click.Path(exists=True, path_type=Path))
@click.argument("outfile", type=click.Path(path_type=Path), required=False, default=None)
def main(infile: Path, outfile: Path | None) -> None:
    """Convert a microscopy image to OME-TIFF format.

    INFILE may be any format supported by slideio (SVS, NDPI, VSI, SCN, ZVI,
    OME-TIFF, standard TIFF/PNG/JPG, DICOM, …).  OUTFILE defaults to the same
    directory and stem as INFILE with a .ome.tiff extension.
    """
    resolved = outfile or _default_outfile(infile)
    convert_generic(infile, resolved)
    click.echo(f"Written: {resolved}")


if __name__ == "__main__":
    main()