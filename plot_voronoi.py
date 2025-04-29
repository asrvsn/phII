from matplotlib import pyplot as plt
import asrvsn_mpl as pt

from lib import *

def fun(stage: str, spheroid: int, path: str):
    fig, ax = plt.subplots()
    boundary = pickle.load(open(f'{path}.boundary.rescaled', 'rb'))
    pt.ax_ellipse(ax, boundary, color='blue')
    cells = pickle.load(open(f'{path}.cell_polygons.rescaled', 'rb'))
    centroids = np.array([p.centroid() for p in cells])
    centroids = centroids[boundary.contains(centroids)]
    vors = pickle.load(open(f'{path}.voronoi.rescaled', 'rb'))
    ax.scatter(centroids[:, 0], centroids[:, 1], color='red', s=1)
    pt.ax_planar_polygons(ax, vors, alpha=0., linecolor='red', linewidth=0.5)
    save_fig(fig, f'{path}.voronoi', ['png'])
    plt.close(fig)

if __name__ == '__main__':
    run_for_each_spheroid_topview(fun)
