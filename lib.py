'''
Base utility functions for the project
'''

from typing import Callable, Tuple, Dict
import numpy as np
import pandas as pd
import os
import glob
import pickle
from matgeo import PlanarPolygon, PlanarPolygonPacking, Ellipse, Plane
from microseg.utils.data import get_voxel_size, load_XY_image
from microseg.pdf.vstack import parse_trim
from matplotlib import pyplot as plt
import asrvsn_mpl as pt
import subprocess
import random
import matplotlib as mpl

DATA_PATH = '../data'
FIGS_PATH = '../figures'
ZENODO_PATH = '../zenodo'
TOPVIEW_STAGES = ['I', 'II', 'III', 'IV', 'S']

def run_for_each_spheroid_topview(fun: Callable, log: bool=True):
    '''
    Run a function over each spheroid in the data/ directory.
    Iteration order is guaranteed to be I.1, I.2, ..., II.1, ...
    '''
    for stage in TOPVIEW_STAGES:
        spheroid = 1
        folder = f'{DATA_PATH}/topview/{stage}/{spheroid}'
        while os.path.exists(folder):
            # Get base CZI file
            czi_file = glob.glob(f'{folder}/*.czi')
            assert len(czi_file) == 1
            path = os.path.splitext(czi_file[0])[0]
            if log:
                print(f'Processing {path}')
            fun(stage, spheroid, path)
            spheroid += 1
            folder = f'{DATA_PATH}/topview/{stage}/{spheroid}'

def get_spheroids_per_stage() -> Dict[str, int]:
    sizes = dict(zip(TOPVIEW_STAGES, [0] * len(TOPVIEW_STAGES)))
    def fun(stage: str, spheroid: int, path: str):
        sizes[stage] += 1
    run_for_each_spheroid_topview(fun, log=False)
    return sizes

def run_for_each_spheroid_crosssection(fun: Callable):
    '''
    Run a function over each spheroid in the data/ directory.
    Iteration order is guaranteed to be I.1, I.2, ..., II.1, ...
    '''
    for stage in ['I', 'II', 'III', 'IV']:
        folder = f'{DATA_PATH}/crosssection/{stage}'
        # Get base TIFF file
        tiff_file = glob.glob(f'{folder}/*.tif')
        assert len(tiff_file) == 1
        path = os.path.splitext(tiff_file[0])[0]
        print(f'Processing {path}')
        fun(stage, path)

''' Plotting '''

def set_preconditions():
        # Preconditions
    random.seed(0)
    np.random.seed(0)
    np.seterr(all='warn')
    # See https://felix11h.github.io/blog/matplotlib-tgheros
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble = '\n'.join([
        r'\usepackage{bm}',    # bold math
        r'\usepackage{tgheros}',    # helvetica font
        r'\usepackage{sansmath}',   # math-font matching  helvetica
        r'\sansmath'                # actually tell tex to use it!
        r'\usepackage{siunitx}',    # micro symbols
        r'\sisetup{detect-all}',    # force siunitx to use the fonts
    ]))


ticks_fs = 16
title_fs = 22
label_fs = 18
means_lw = 2 # Line width for mean bars
upperchars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
group_lw = 2 # Line width for group plots
ind_lw = 1 # Line width for individual plots
scatter_alpha = 0.5

stage_colors = { 
    'I': 'blue', # map_color_01(0, cc.bgy), # Corresponding to wall-clock time, "green" for algae :)
    'II': 'red', # map_color_01(15/48, cc.bgy),
    'III': 'magenta', # map_color_01(21/48, cc.bgy),
    'IV': 'darkgreen', # map_color_01(37/48, cc.bgy),
    'S': 'darkorange',
}

def save_fig(fig: plt.Figure, pathname: str, exts: list[str], dpi=300, compressed_dpi=100, jpeg_quality=85, autorasterize=True, trim=""):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(pathname), exist_ok=True)
    for ext in exts:
        path = f'{pathname}.{ext}'
        if ext == 'pdf':
            if autorasterize:
                pt.rasterize_fig(fig)
            path_uncompressed = f'{pathname}.uncompressed.{ext}'
            fig.savefig(path_uncompressed, bbox_inches='tight', dpi=dpi)
            print(f'Saved uncompressed figure to {path_uncompressed}')
            # Use ghostscript to optimize PDF
            gs_command = [
                "gs",
                "-sDEVICE=pdfwrite",  # Output device
                "-dCompatibilityLevel=1.5",  # PDF version compatibility
                f"-dPDFSETTINGS=/printer",  # Adjust settings for print-quality compression
                f"-dAutoFilterColorImages=true",
                # f"-dColorImageFilter=/DCTEncode",  # Force JPEG compression
                f"-dColorImageDownsampleType=/Bicubic",  # Downsample method
                f"-dColorImageResolution={compressed_dpi}",  # Downsample resolution
                f"-dAutoFilterGrayImages=true",
                # f"-dGrayImageFilter=/DCTEncode",  # Force JPEG compression
                f"-dGrayImageDownsampleType=/Bicubic",  # Downsample method
                f"-dGrayImageResolution={compressed_dpi}",  # Downsample resolution
                f"-dJPEGQ={jpeg_quality}",  # JPEG compression quality
                f"-dMonoImageFilter=/CCITTFaxEncode",
                f"-dUseFlateCompression=true",
                f"-dCompressPages=true",
                f"-dDiscardDocumentStruct=true",
                f"-dDiscardMetadata=true",
                f"-dSubsetFonts=true",
                f"-dEmbedAllFonts=true",
                "-dNOPAUSE",  # No pause between pages
                "-dBATCH",  # Batch mode (exit when done)
                "-dQUIET",  # Suppress messages
                f"-sOutputFile={path}",  # Output file
                path_uncompressed  # Input file
            ]
            subprocess.run(gs_command, check=True)
            print(f'Saved compressed figure to {path}')
            if trim:
                l, b, r, t = parse_trim(trim) # latex-style command
                crop_cmd = [
                    'pdfcrop',
                    '--margins',
                    f'{-l} {-t} {-r} {-b}',
                    path, path
                ]
                with open(os.devnull, 'w') as devnull:
                    subprocess.run(crop_cmd, check=True, stdout=devnull, stderr=devnull)
                print(f'Trimmed figure to {path}')
        else:
            fig.savefig(path, bbox_inches='tight', dpi=dpi)
        print(f'Saved figure to {path}')

def pad_left_ax(ax, x_frac: float=0.1):
    x_min, x_max = ax.get_xlim()
    if ax.get_xscale() == 'log':
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        log_range = log_max - log_min
        new_log_min = log_min - (log_range * x_frac)
        new_x_min = 10**new_log_min
    else:
        x_range = x_max - x_min
        new_x_min = x_min - (x_range * x_frac)
    ax.set_xlim(new_x_min, x_max)

def pad_right_ax(ax, x_frac: float=0.1):
    x_min, x_max = ax.get_xlim()
    if ax.get_xscale() == 'log':
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        log_range = log_max - log_min
        new_log_max = log_max + (log_range * x_frac)
        new_x_max = 10**new_log_max
    else:
        x_range = x_max - x_min
        new_x_max = x_max + (x_range * x_frac)
    ax.set_xlim(x_min, new_x_max)

def pad_top_ax(ax, y_frac: float=0.1):
    y_min, y_max = ax.get_ylim()
    if ax.get_yscale() == 'log':
        log_min = np.log10(y_min)
        log_max = np.log10(y_max)
        log_range = log_max - log_min
        new_log_max = log_max + (log_range * y_frac)
        new_y_max = 10**new_log_max
    else:
        y_range = y_max - y_min
        new_y_max = y_max + (y_range * y_frac)
    ax.set_ylim(y_min, new_y_max)

def flatten_list(l: list[list]) -> list:
    flattened = []
    for item in l:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened