import os
import shutil
from typing import Literal

import click
import hdf5plugin
import anndata as ad
import scipy.sparse as sp
import numpy as np
import tempfile


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(), required=False)
@click.option('--compression', type=click.Choice(['gzip', 'lzf', 'zstd']), default='zstd',
              help='Compression algorithm to use for the output .h5ad file.')
def main(input_file: str, output_file: str = None, compression: Literal["gzip", "lzf", "zstd"] = "gzip"):
    adata = ad.read_h5ad(input_file)  # Read the input .h5ad file

    # for each layer, if not a sparse matrix, convert to sparse
    click.echo(
        "Converting dense matrices to sparse..."
    )

    # Check main matrix
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    # Check layers
    for layer in adata.layers:
        if not sp.issparse(adata.layers[layer]):
            adata.layers[layer] = sp.csr_matrix(adata.layers[layer])
    # Check obsm
    for key in adata.obsm:
        if not sp.issparse(adata.obsm[key]) and isinstance(adata.obsm[key], np.ndarray):
            adata.obsm[key] = sp.csr_matrix(adata.obsm[key])
    # Check varm
    for key in adata.varm:
        if not sp.issparse(adata.varm[key]) and isinstance(adata.varm[key], np.ndarray):
            adata.varm[key] = sp.csr_matrix(adata.varm[key])
    # Check obsp
    for key in adata.obsp:
        if not sp.issparse(adata.obsp[key]):
            adata.obsp[key] = sp.csr_matrix(adata.obsp[key])
    # Check varp
    for key in adata.varp:
        if not sp.issparse(adata.varp[key]):
            adata.varp[key] = sp.csr_matrix(adata.varp[key])
    # Check .raw
    if adata.raw is not None:
        if not sp.issparse(adata.raw.X):
            adata.raw.X = sp.csr_matrix(adata.raw.X)

    # Finally, write the output .h5ad file with compression
    click.echo(
        f"Writing compressed .h5ad file to {output_file} with {compression} compression..."
    )

    is_same = output_file is None or (input_file == output_file)
    with tempfile.NamedTemporaryFile() as tmp_file:
        temp_output_file = tmp_file.name if is_same else output_file
        if compression != "zstd":
            adata.write_h5ad(temp_output_file, compression=compression)  # Use built-in compression
        else:
            # Use zstd compression via hdf5plugin
            adata.write_h5ad(temp_output_file,
                             compression=hdf5plugin.FILTERS['zstd'],
                             compression_opts=hdf5plugin.Zstd(clevel=5).filter_options
                             )

        # If overwriting, move temp file to original location
        if is_same:
            os.remove(input_file)
            shutil.copy(temp_output_file, input_file)
        else:
            shutil.copy(temp_output_file, input_file)

    click.echo(
        f"Compression completed, h5ad file saved to: {output_file if output_file else input_file}"
    )


if __name__ == '__main__':
    main()