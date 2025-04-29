'''
Functions for plotting all spheroids
'''

from lib import *
from compute_spheroid_metrics import SpheroidMetricContext
from microseg.utils.data import load_stack
def plot_ap(
        ax: plt.Axes, 
        path: str,
        linewidth=3, 
        gonidia=True, 
        labels=False, 
        label_color='black', 
        fontsize=22
    ):
    boundary = pickle.load(open(f'{path}.boundary', 'rb'))
    pt.ax_ellipse(ax, boundary, color='white', axes=False, linewidth=linewidth)
    v = boundary.v 
    anterior_offspring = pickle.load(open(f'{path}.offspring.anterior', 'rb'))
    coms = np.array([p.centroid() for p in anterior_offspring])
    axis = SpheroidMetricContext(boundary, anterior_offspring).get_ap_axis()
    if gonidia:
        # Plot gonidia 
        ax.scatter(coms[:,0], coms[:,1], marker='o', s=100, color='lightgreen', zorder=10)
        # Plot line thru gonidia
        gon_line = Plane.fit_l2(coms) # Fit line thru gonidia centers of mass
        gon_pts = boundary.plane_intersection(gon_line)
        assert gon_pts.shape[0] == 2
        ax.plot(gon_pts[:,0], gon_pts[:,1], color='lightgreen', zorder=10, linewidth=linewidth)
    w = v + axis
    u = v - axis
    ax.plot([u[0], w[0]], [u[1], w[1]], color='yellow', zorder=10, linewidth=linewidth)
    if labels:
        for lbl, sign in [('A', 1), ('P', -1)]:
            v_ = v + axis * 1.05 * sign
            ax.text(v_[0], v_[1], r'\textbf{'+lbl+r'}', fontsize=fontsize, ha='center', va='center', color=label_color)       

def plot_segmentation(
        ax: plt.Axes,
        stage: str,
        spheroid: int,
        path: str,
        with_compartments: bool = True,
        with_hull_pts: bool = True,
        **kwargs,
    ):
    voxsize = get_voxel_size(f'{path}.czi', fmt='XY')
    # 1. brightfield
    ax.axis('off')
    brightfield = np.load(f'{path}.zproj.npy')[2] 
    pt.ax_im_gray(ax, brightfield)
    # 2. compartments
    if with_compartments:
        compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
        areas = np.array([p.area() for p in compartments])
        compartments = [p.rescale(1/voxsize) for p in compartments]
        pt.ax_planar_polygons(ax, compartments, cmap=lambda _: pt.map_colors(areas, 'continuous'), linewidth=0)
    # 3. boundary
    boundary = pickle.load(open(f'{path}.boundary', 'rb'))
    pt.ax_ellipse(ax, boundary, major_axis_color='white', minor_axis_color='none', axes=True, axes_linewidth=3)
    # 4. hull points
    if with_hull_pts:
        verts_path = f'{path}.vertices'
        if os.path.exists(verts_path):
            vertices = np.loadtxt(verts_path)
            vertices = vertices[:, :2] / voxsize
            ax.scatter(vertices[:, 0], vertices[:, 1], color='#00BFFF', s=2)
    # 5. AP axis
    plot_ap(ax, path, **kwargs)
    # 6. Formatting
    pt.label_ax(ax, f'{stage}.{spheroid}', fontsize=18)
    ax.set_xlim(0, brightfield.shape[1])
    ax.set_ylim(brightfield.shape[0], 0)
    assert np.isclose(voxsize[0], voxsize[1]), f'Got voxel size {voxsize} for {path}'
    pt.ax_scale_bar(ax, 100, upp=voxsize[0], color='black', units=r'$\mu$m', fontsize=16, label=(stage == 'I' and spheroid == 1))

def make_segmentation_layout():
    n_rows = max(get_spheroids_per_stage().values())
    n_cols = len(TOPVIEW_STAGES)
    layout = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    return layout

def plot_segmentations(**kwargs):
    '''
    Main diagnostic plot for segmentations
    '''
    layout = make_segmentation_layout()
    fig, axs = pt.default_mosaic(layout)

    def fun(stage: str, spheroid: int, path: str):
        i = spheroid - 1
        j = TOPVIEW_STAGES.index(stage)
        ax = axs[layout[i, j]]
        if spheroid == 0:
            pt.title_ax(ax, f'Stage {stage}')
        plot_segmentation(ax, stage, spheroid, path, **kwargs)
        ax._plotted_seg = True

    run_for_each_spheroid_topview(fun, log=False)

    for ax in axs.values():
        if not hasattr(ax, '_plotted_seg'):
            ax.axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, axs
    
def plot_offspring_segmentations():
    layout = make_segmentation_layout()
    fig, axs = pt.default_mosaic(layout)

    def fun(stage: str, spheroid: int, path: str):
        i = spheroid - 1
        j = TOPVIEW_STAGES.index(stage)
        ax = axs[layout[i, j]]
        if i == 0:
            pt.title_ax(ax, f'Stage {stage}')
        ax.axis('off')
        # 0. load offspring
        offspring = pickle.load(open(f'{path}.offspring', 'rb'))
        # 1. find z-plane from brightfield with most offspring
        zstack = load_stack(f'{path}.czi')[:, :, :, 2].astype(np.float32) # ZYX
        z = max(enumerate(offspring), key=lambda x: len(x[1]))[0]
        zimg = zstack[z].transpose() # transpose to be consistent with z-projection in other figures which is transposed
        pt.ax_im_gray(ax, zimg)
        # 2. offspring
        offs = [p.transpose() for p in offspring[z]] # transpose to be consistent with z-projection in other figures which is transposed
        pt.ax_planar_polygons(ax, offs, alpha=0., linecolor='lightgreen', linewidth=1.5)
        # Format
        voxsize = get_voxel_size(f'{path}.czi', fmt='XY')
        pt.label_ax(ax, f'{stage}.{spheroid}', fontsize=18)
        ax.set_xlim(0, zimg.shape[1])
        ax.set_ylim(zimg.shape[0], 0)
        assert np.isclose(voxsize[0], voxsize[1]), f'Got voxel size {voxsize} for {path}'
        pt.ax_scale_bar(ax, 100, upp=voxsize[0], color='black', units=r'$\mu$m', fontsize=16, label=(stage == 'I' and spheroid == 1))
        ax._plotted_seg = True

    run_for_each_spheroid_topview(fun, log=False)

    for ax in axs.values():
        if not hasattr(ax, '_plotted_seg'):
            ax.axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, axs
