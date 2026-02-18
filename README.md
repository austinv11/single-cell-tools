# Single Cell Tools

This repository contains a collection of tools for basic processing of single cell datasets.

## Tools Included
- compress_h5ad: Compress h5ad files to reduce file size.
- attach_to_notebook: Attach to a running Jupyter notebook to view its output in the terminal.

## Installation

You need a C compiler (`gcc` on Linux, Xcode Command Line Tools on macOS, or MSVC on Windows) — Nuitka uses it to compile the tools to native binaries during install.

Install from GitHub:

```bash
pip install git+https://github.com/austinv11/single-cell-tools.git
```

Or with uv:

```bash
uv pip install git+https://github.com/austinv11/single-cell-tools.git
```

> The first install takes a few minutes while Nuitka compiles the tools. Subsequent uses are instant.

## Usage:

```bash
compress_h5ad input_file.h5ad  # Inline compression
```

```bash
attach_to_notebook my_notebook.ipynb  # Attach to a Jupyter notebook to view execution results inline
```