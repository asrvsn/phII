'''
Functions for plotting all spheroids
'''

from lib import *
import cv2
from compute_spheroid_metrics import SpheroidMetricContext
from microseg.utils.data import load_stack, get_voxel_size
from matgeo import Sphere, Ellipsoid

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

def plot_foam():
    full_paths = [os.path.join(DATA_PATH, f) for f in [
        'topview/II/closeups/Ph2_cYFP_top_view_1_complete_Stack_cropped.tif',
        'topview/IV/closeups/MAX_231116_5_1_phII_2-3d_stack (RGB).tif',
    ]]
    crop_paths = [os.path.join(DATA_PATH, f) for f in [
        'topview/II/closeups/Ph2_cYFP_top_view_1_complete_Stack_detail_2.tif',
        'topview/IV/closeups/231116_5_1_phII_2-3d_substack (RGB).tif',
    ]]
    full_imgs = [load_stack(path) for path in full_paths]
    crop_imgs = [load_stack(path) for path in crop_paths]
    scales = [get_voxel_size(path, fmt='XY')[1] for path in full_paths]
    pfs = []

    def find_crop_pos(full_img, crop_img):
        ''' Match the crop to the original by a search '''
        # Z, C must be the same shape
        assert full_img.shape[0] == crop_img.shape[0]
        assert full_img.shape[-1] == crop_img.shape[-1]
        # X, Y must be contained
        assert full_img.shape[1] >= crop_img.shape[1]
        assert full_img.shape[2] >= crop_img.shape[2]
        # Find crop 
        full_afl, crop_afl = full_img[2], crop_img[2]
        result = cv2.matchTemplate(full_afl, crop_afl, cv2.TM_CCOEFF_NORMED)
        _, __, ___, max_loc = cv2.minMaxLoc(result)
        return max_loc
        
    # pad = .3 # Percentage adding on either side
    pad = 0 # Percentage adding on either side
    cell_color_overlay = 'white'
    comp_color_overlay = 'coral'
    cell_color = 'red'
    comp_color = 'coral'
    vor_color = 'lightsteelblue'
    layout = [
        [str((i, j)) for j in range(4)] for i in range(2)
    ]
    fig, axs = plt.subplot_mosaic(layout, figsize=(4 * len(layout[0]), 4 * len(layout)))
    for i, (full_img, crop_img) in enumerate(zip(full_imgs, crop_imgs)):
        full_name = os.path.splitext(full_paths[i])[0]
        cells_path = f'{full_name}.cells'
        compartments_path = f'{full_name}.compartments'
        assert os.path.isfile(cells_path) and os.path.isfile(compartments_path)
        cells = pickle.load(open(cells_path, 'rb'))
        compartments = pickle.load(open(compartments_path, 'rb'))
        crop_pos = find_crop_pos(full_img, crop_img)
        bdry = PlanarPolygon.from_shape(full_img.shape[1:-1])
        cell_coms = np.array([c.v for c in cells])
        vors = bdry.voronoi_tessellate(cell_coms).polygons
        sub_bdry = PlanarPolygon.from_shape(crop_img.shape[1:-1]) + crop_pos[:2]
        sub_compartments = flatten_list([sub_bdry.intersection(p) for p in compartments])
        sub_packing = PlanarPolygonPacking(sub_bdry, sub_compartments)
        pfs.append(sub_packing.packing_fraction())
        set_label = lambda ax, j, color: pt.label_ax(ax, upperchars[i]+str(j+1), color=color)
        # compartments = sub_compartments
        print(f'Average aspect ratio: {np.mean([c.aspect_ratio() for c in compartments])}')
        print(f'Average shape index: {np.mean([c.shape_index() for c in compartments])}')
        # 1. Chl + YFP
        ax = axs[str((i, 0))]
        set_label(ax, 0, 'white')
        comb_img = np.max([full_img[1], full_img[2]], axis=0)
        ax.imshow(comb_img)
        # 2. Chl + YfP + Segments
        ax = axs[str((i, 1))]
        set_label(ax, 1, 'white')
        ax.imshow(comb_img)
        pt.ax_ellipses(ax, cells, color=cell_color_overlay, linewidth=2)
        # pt.ax_planar_polygons(ax, vors, alpha=0., linecolor=vor_color, linewidth=2)
        pt.ax_planar_polygons(ax, compartments, alpha=0., linecolor=comp_color_overlay, linewidth=1.5, linestyle='--')
        # 3. Segments 
        ax = axs[str((i, 2))]
        set_label(ax, 2, 'black')
        pt.ax_ellipses(ax, cells, color=cell_color, linewidth=1)
        # pt.ax_planar_polygons(ax, vors, alpha=0., linecolor=vor_color, linewidth=1)
        pt.ax_planar_polygons(ax, compartments, alpha=0., linecolor=comp_color, linewidth=1, linestyle='--')
        # # 4. Trans-PMT
        # ax = axs[str((i, 3))]
        # ax.imshow(full_img[0])
        # 4. Segments + Voronoi
        ax = axs[str((i, 3))]
        set_label(ax, 3, 'black')
        pt.ax_ellipses(ax, cells, color=cell_color, linewidth=1)
        pt.ax_planar_polygons(ax, vors, alpha=0., linecolor=vor_color, linewidth=1)
        pt.ax_planar_polygons(ax, compartments, alpha=0., linecolor=comp_color, linewidth=1, linestyle='--')
        # Common formatting
        for j in range(4):
            ax = axs[str((i, j))]
            epsx = pad * crop_img.shape[1]
            epsy = pad * crop_img.shape[2]
            ax.set_xlim(crop_pos[0] - epsx, crop_pos[0] + crop_img.shape[1] + epsx)
            ax.set_ylim(crop_pos[1] + crop_img.shape[2] + epsy, crop_pos[1] - epsy) # Y axis flipped by mpl
            if pad > 0:
                # Plot rect 
                pt.ax_planar_polygons(ax, [sub_bdry], alpha=0., linecolor='white', linewidth=1, zorder=10)
            pt.ax_scale_bar(ax, 25, color=('white' if j < 2 else 'black'), upp=scales[i], units=r'$\bm{\mu}$m', label=(j==0), fontsize=13)
            ax.axis('off')

    pt.left_title_ax(axs[str((0, 0))], f'stage II ($\phi={pfs[0]:.2f}$)', fontsize=14, offset=-0.05)
    pt.left_title_ax(axs[str((1, 0))], f'stage IV ($\phi={pfs[1]:.2f}$)', fontsize=14, offset=-0.05)
    fig.subplots_adjust(wspace=.01, hspace=.01)
    return fig, axs

def plot_schematic(stage: str, spheroid: int):
    def load_data(stage: str, spheroid: int, path: str):
        voxsize = get_voxel_size(f'{path}.czi', fmt='XY')
        brightfield = np.load(f'{path}.zproj.npy')[2]
        compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
        cells = pickle.load(open(f'{path}.cell_polygons.rescaled.matched', 'rb'))
        vors = pickle.load(open(f'{path}.voronoi.rescaled.matched', 'rb'))
        boundary = pickle.load(open(f'{path}.boundary', 'rb'))
        
        # Use non-rescaled versions for display (already in pixel coordinates)
        display_compartments = pickle.load(open(f'{path}.compartment_polygons.matched', 'rb'))
        display_cells = pickle.load(open(f'{path}.cell_polygons.matched', 'rb'))
        # Manual rescale for voronoi (only rescaled version exists)
        display_vors = [p.rescale(1/voxsize) for p in vors]
        
        return {
            'voxsize': voxsize,
            'brightfield': brightfield,
            'compartments': compartments,
            'cells': cells,
            'vors': vors,
            'boundary': boundary,
            'display_compartments': display_compartments,
            'display_cells': display_cells,
            'display_vors': display_vors,
            'path': path
        }
    
    data = get_from_spheroid(stage, spheroid, load_data, log=False)
    
    layout = [
        ['z', 'z', 'z', 'a', 'b'],
        ['z', 'z', 'z', 'c', 'd'],
        ['z', 'z', 'z', 'e', 'f'],
    ]
    fig, axs = plt.subplot_mosaic(layout, figsize=(4 * len(layout[0]), 4 * len(layout)))

    for k, ax in axs.items():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    scm_lw = 3
    cell_color = 'greenyellow'
    comp_color = 'gold'
    my_scale_fs = int(1.7 * title_fs)
    my_label_fs = int(2.8 * title_fs)

    # Pick a specific polygon for demonstration
    disp_idx = min(50, len(data['display_compartments']) - 1)
    disp_poly = data['display_compartments'][disp_idx]
    disp_cell = Sphere.from_poly(data['display_cells'][disp_idx])
    print(f'Cell radius (um): {data["cells"][disp_idx].circular_radius()}')
    imgx = data['brightfield'].shape[1]

    def show_seg(ax, title, show_poly=True):
        areas = np.array([p.area() for p in data['compartments']])
        pt.ax_im_gray(ax, data['brightfield'])
        pt.ax_planar_polygons(ax, data['display_compartments'], cmap=lambda _: pt.map_colors(areas, 'continuous'), linewidth=0)
        if show_poly:
            pt.ax_ppoly(ax, disp_poly, alpha=0., linewidth=2, linecolor=comp_color)

    def zoom_poly(ax, sctx=False):
        x0, y0, x1, y1 = disp_poly.bounding_box()
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dl = max(x1 - x0, y1 - y0)
        ax.set_xlim(xm-dl/2, xm+dl/2)
        ax.set_ylim(ym+dl/2, ym-dl/2)
        pt.ax_scale_bar(ax, 10, upp=data['voxsize'][0], color='black', units=r'$\bm{\mu}$m', fontsize=my_scale_fs, linewidth=8, label=sctx, text_pad=0.08)

    def callout_poly(ax):
        x0, y0, x1, y1 = disp_poly.bounding_box()
        wh = max(x1 - x0, y1 - y0) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pt.ax_rect(ax, np.array([cx, cy]), (wh, wh), linewidth=4, color='black')

    # Volvox
    ax = axs['z']
    show_seg(ax, 'trans-PMT + PhII segments', show_poly=False)
    plot_ap(ax, data['path'], linewidth=means_lw*2, labels=True, fontsize=int(1.5*title_fs))
    callout_poly(ax)
    pt.label_ax(ax, 0, fontsize=my_label_fs, xy=(0.02, 0.98))
    assert np.isclose(data['voxsize'][0], data['voxsize'][1]), 'Asymmetric resolution'
    crop = 180
    ax.set_xlim(crop, imgx)
    ax.set_ylim(data['brightfield'].shape[0]-crop, 0)
    pt.ax_scale_bar(ax, 100, upp=data['voxsize'][0], color='black', units=r'$\bm{\mu}$m', fontsize=my_scale_fs, linewidth=8, text_pad=0.06)

    # Original
    ax = axs['a']
    show_seg(ax, r'$a_{\rm cz3}, a_{\rm cell}$')
    pt.ax_ellipse(ax, disp_cell, color=cell_color, linewidth=4)
    ax.scatter(disp_cell.v[0], disp_cell.v[1], color=cell_color, s=600, marker='*', zorder=3)
    zoom_poly(ax, sctx=True)
    pt.label_ax(ax, 1, fontsize=my_label_fs)

    # Aspect ratio
    ax = axs['b']
    show_seg(ax, r'$\alpha$')
    zoom_poly(ax)
    pt.label_ax(ax, 2, fontsize=my_label_fs)
    poly_ell = Ellipsoid.from_poly(disp_poly)
    pt.ax_ellipse(ax, poly_ell, color='cyan', linewidth=scm_lw, axes=True, axes_linewidth=scm_lw)

    # Circularity
    ax = axs['c']
    show_seg(ax, r'$q$')
    zoom_poly(ax)
    poly_circ = Sphere.from_poly(disp_poly)
    pt.ax_ellipse(ax, poly_circ, color='cyan', linewidth=scm_lw)
    pt.label_ax(ax, 3, fontsize=my_label_fs)

    # Offset
    ax = axs['d']
    show_seg(ax, r'$r$')
    zoom_poly(ax)
    pt.ax_ellipse(ax, disp_cell, color=cell_color, linewidth=4)
    pt.label_ax(ax, 4, fontsize=my_label_fs)
    v1, v2 = disp_poly.centroid(), disp_cell.v
    ax.scatter(v1[0], v1[1], color=comp_color, s=600, marker='*', zorder=3)
    ax.scatter(v2[0], v2[1], color=cell_color, s=600, marker='*', zorder=3)

    # Whitened
    ax = axs['e']
    show_seg(ax, r'$\tilde{r}$')
    zoom_poly(ax)
    poly_w, T = disp_poly.whiten(return_W=True)
    poly_w = poly_w.match_area(disp_poly.area())
    pt.ax_ppoly(ax, poly_w, alpha=0., linewidth=scm_lw, linecolor='cyan')
    pt.label_ax(ax, 5, fontsize=my_label_fs)

    # Voronoi error
    ax = axs['f']
    show_seg(ax, r'$e_V$')
    pt.ax_planar_polygons(ax, data['display_vors'], alpha=0., linewidth=scm_lw, linecolor='white')
    zoom_poly(ax)
    pt.label_ax(ax, 6, fontsize=my_label_fs)

    fig.subplots_adjust(left=0., right=1., bottom=0.0, top=1., wspace=0.02, hspace=0.02)
    return fig, axs
