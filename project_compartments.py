'''
Correct cell, compartment and voronoi polygons for systematic elastic distortion

The resulting polygons will still be embedded in the plane for convenience, but their
positions relative to one another will no longer make sense. DO NOT USE THEM for any
intersections, unions, etc.

Triplets (cell, compartment, voronoi) will be consistent in the same basis and can be used for 
offset, intersection, union, calculations as one does in compute_metrics.py.
'''

from lib import *
from scipy.spatial import ConvexHull
import matplotlib
import matplotlib.colors
from mpl_toolkits.mplot3d import art3d

def run():
    def fun(stage: str, spheroid: int, path: str):
        ell_path = f'{path}.ellipsoid'
        if os.path.exists(ell_path):
            ell = pickle.load(open(ell_path, 'rb'))
            compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
            cells = pickle.load(open(f'{path}.cell_polygons.rescaled.matched', 'rb'))
            vors = pickle.load(open(f'{path}.voronoi.rescaled.matched', 'rb'))

            # Translate ellipsoid so it's above the XY plane
            ell.v[2] = ell.get_radii().max() * 1.2

            # Distortion correct by projecting to ellipsoid
            compartments_proj = [ell.project_poly_z(p) for p in compartments]
            # Use the compartment plane basis for other projections
            planes = [p.plane for p in compartments_proj]
            cells_proj = [ell.project_poly_z(p, plane=plane) for p, plane in zip(cells, planes)]

            # Save
            pickle.dump(cells_proj, open(f'{path}.cell_polygons.rescaled.matched.projected', 'wb'))
            pickle.dump(compartments_proj, open(f'{path}.compartment_polygons.rescaled.matched.projected', 'wb'))
    
    run_for_each_spheroid_topview(fun)

def render_projection(ax, stage: str, spheroid: int):
    assert hasattr(ax, 'get_zlim'), 'ax must be a 3D axis'
    folder = f'{DATA_PATH}/topview/{stage}/{spheroid}'
    # Get base CZI file
    czi_file = glob.glob(f'{folder}/*.czi')
    assert len(czi_file) == 1
    path = os.path.splitext(czi_file[0])[0]
    vertices = np.loadtxt(f'{path}.vertices')
    ell = pickle.load(open(f'{path}.ellipsoid', 'rb'))
    # Translate ellipsoid so it's above the XY plane
    zdelta = ell.get_radii().max() * 1.2 - ell.v[2]
    ell.v[2] += zdelta
    vertices[:, 2] += zdelta
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=2, marker='x')
    # Plot ellipsoidal patch intersecting the vertices
    n = 300
    us, vs = np.mgrid[0:2*np.pi:2*n*1j, 0:np.pi:n*1j]
    x = np.cos(us)*np.sin(vs)
    y = np.sin(us)*np.sin(vs)
    z = np.cos(vs)
    X = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    X = ell.map_sphere(X)
    zmax = vertices[:, 2].max() * 1.01
    X = X[X[:, 2] <= zmax]
    chull = ConvexHull(X)
    lightsource = matplotlib.colors.LightSource(
        azdeg=225, 
        altdeg=-45,
        hsv_min_val=0.2,  
        hsv_max_val=1.0, 
        hsv_min_sat=0.1,  
        hsv_max_sat=0.5
    )
    ax.plot_trisurf(
        X[chull.vertices, 0], X[chull.vertices, 1], X[chull.vertices, 2], 
        color='green', alpha=0.5, shade=True, lightsource=lightsource
    )
    # Set the view to look from below
    ax.view_init(elev=-30, azim=30)
    ax.dist = 6.5
    # Make pretty
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.grid(False)
    # Plot projections
    proj_polys = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched.projected', 'rb'))
    art_kwargs = dict(
        edgecolor='black',
        alpha=0.5,
        lightsource=lightsource,
        zorder=0, 
        linewidth=0.5,
        linestyle='--',
    )
    pc = art3d.Poly3DCollection([
        p.vertices_nd for p in proj_polys
    ], **art_kwargs)
    ax.add_collection(pc)
    
    # Add orientation widget in bottom-left corner
    pt.add_xyz_axes_widget(ax, scale=0.1, pos_2d=(-0.09, 0.33))

if __name__ == '__main__':
    run()

