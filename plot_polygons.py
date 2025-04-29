from matplotlib import pyplot as plt
import asrvsn_mpl as pt

from lib import *

def fun(stage: str, spheroid: int, path: str):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    zproj = np.load(f'{path}.zproj.npy')
    boundary = pickle.load(open(f'{path}.boundary', 'rb'))
    for j in [0, 1, 2]:
        for i in [0, 1]:
            axs[i, j].imshow(zproj[j], cmap='gray')
            axs[i, j].set_axis_off()
            pt.ax_ellipse(axs[i, j], boundary, color='blue')

    voxsize = get_voxel_size(f'{path}.czi', fmt='XY')

    pt.left_title_ax(axs[0, 0], 'All')
    cells = [p.rescale(1/voxsize) for p in pickle.load(open(f'{path}.cell_polygons.rescaled', 'rb'))]
    compartments = [p.rescale(1/voxsize) for p in pickle.load(open(f'{path}.compartment_polygons.rescaled', 'rb'))]
    for j in [0, 1, 2]:
        if j in [1, 2]:
            pt.ax_planar_polygons(axs[0, j], cells, alpha=0., linecolor='red', linewidth=0.5)
        if j in [0, 2]:
            pt.ax_planar_polygons(axs[0, j], compartments, alpha=0., linecolor='orange', linewidth=0.5)

    pt.left_title_ax(axs[1, 0], 'Matched')
    cells = [p.rescale(1/voxsize) for p in pickle.load(open(f'{path}.cell_polygons.rescaled.matched', 'rb'))]
    compartments = [p.rescale(1/voxsize) for p in pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))]
    for j in [0, 1, 2]:
        if j in [1, 2]:
            pt.ax_planar_polygons(axs[1, j], cells, alpha=0., linecolor='red', linewidth=0.5)
        if j in [0, 2]:
            pt.ax_planar_polygons(axs[1, j], compartments, alpha=0., linecolor='orange', linewidth=0.5)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    save_fig(fig, f'{path}.polygons', ['png'])
    plt.close(fig)

if __name__ == '__main__':
    run_for_each_spheroid_topview(fun)
