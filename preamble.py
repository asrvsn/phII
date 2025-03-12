import random
import subprocess
import scipy.stats as stats
import numpy as np
import shapely
import shapely.geometry
import shapely.geometry.polygon
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

from ph2_groups import *
from asrvsn_mpl.rvs import *
from microseg.utils.colors import *
import microseg.utils.sigfigs as sigfigs
from matgeo import Ellipsoid, Sphere

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

# Constants
log_hist = True
bins_hist = 50
ap_xticks = np.linspace(-50, 50, Ph2Segmentation.ap_bins + 1) # Percent of major axis length
ap_xvalues = (ap_xticks[:-1] + ap_xticks[1:]) / 2 # Center of each bin
xlim_hist = 4 # Limit of x-axis for nondimensional gamma plots
gam_bins = np.linspace(0, xlim_hist, bins_hist + 1) # Bins for nondimensional gamma plots
legend_loc = 'upper left'
ind_lw = 1 # Line width for individual plots
ind_alpha = 0.6 # Alpha for individual plots
group_lw = 2 # Line width for group plots
means_lw = 2 # Line width for mean bars
err_alpha = 0.3 # Alpha for error bars
meanplot_bins = 50
hist_legend_loc = 'upper right'
save_exts = ['png', 'pdf']
label_fs = 18
title_fs = 22
ticks_fs = 16

chars = 'abcdefghijklmnopqrstuvwxyz'
upperchars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
scatter_size = 1
scatter_alpha = 0.5
n_contours = 5
contour_alpha = 0.4

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

data_path = os.path.abspath('../data')
figures_path = os.path.abspath('../figures')
recompute_segs = False # Set to true if pickled objects don't load as expected on your machine

def save_fig(fig: plt.Figure, ph2_name: str, exts=save_exts, dpi=300, compressed_dpi=100, jpeg_quality=85, autorasterize=True):
    for ext in exts:
        if ext == 'pdf':
            if autorasterize:
                pt.rasterize_fig(fig)
            path_uncompressed = os.path.join(figures_path, f'{ph2_name}.uncompressed.{ext}')
            path = os.path.join(figures_path, f'{ph2_name}.{ext}')
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
        else:
            path = os.path.join(figures_path, f'{ph2_name}.{ext}')
            fig.savefig(path, bbox_inches='tight', dpi=dpi)
        print(f'Saved figure to {path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process segmentation data.')
    parser.add_argument('filepath', type=str, help='Path to the input file (mandatory)')
    args = parser.parse_args()
    fp = args.filepath
    assert os.path.splitext(fp)[1] == '.seg_ph2', f'File must have .seg_ph2 extension: {fp}'

    seg = pickle.load(open(fp, 'rb'))
    # seg.render_ellipsoid_projection()
    seg.render_chull_projection()