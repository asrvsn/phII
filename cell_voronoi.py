'''
Compute Voronoi polygons for each cell
'''

from lib import *
from scipy.spatial.distance import cdist

def compute_voronoi_polygons(self):
    # Compute Voronoi restricted to the outline (rescaled to dimensionless units)
    raw_polys = mask_to_polygons(self.mask[1], rdp_eps=self.rdp_eps, erode=self.mask_erode, dilate=self.mask_dilate)
    self.all_display_cell_polygons = [PlanarPolygon(p) for p in raw_polys if len(p) >= 3]
    self.all_cell_polygons = [p.set_res(*self.upp) for p in self.all_display_cell_polygons]
    self.all_display_cell_coms = np.array([p.centroid() for p in self.all_display_cell_polygons])
    self.all_cell_coms = np.array([p.centroid() for p in self.all_cell_polygons])
    boundary = PlanarPolygon.from_shape(self.combined.shape)
    vor_polygons = boundary.voronoi_tessellate(self.all_display_cell_coms).polygons
    Nv = len(vor_polygons)
    assert Nv == len(self.all_display_cell_polygons), 'Must have a Voronoi polygon for each cell'
    # Re-order vor_polygons so it is 1-1 with self.polygons, self.cell_polygons on the first N indices
    N = len(self.polygons)
    assert Nv >= N, 'Must have at least as many Voronoi polygons as compartments'
    dists = cdist(self.display_cell_coms, self.all_display_cell_coms)
    idxs = np.argmin(dists, axis=1)
    assert np.unique(idxs).size == N, 'Must have a unique closest cell for each vor cell'
    assert np.allclose(0, dists.min(axis=1)), 'Must have a zero-distance cell for each vor cell'
    rest_idxs = np.setdiff1d(np.arange(Nv), idxs)
    self.display_vor_polygons = \
        [vor_polygons[i] for i in idxs] + \
        [vor_polygons[i] for i in rest_idxs]
    # Rescale to proper units
    self.vor_polygons = [p.set_res(*self.upp) for p in self.display_vor_polygons]
    # Final sanity check: all vor should intersect with corresponding ph2
    assert all([p.intersects(p_) for p, p_ in zip(self.display_polygons, self.display_vor_polygons)]), 'All PhII polygons must intersect with corresponding Voronoi polygons'
    assert all([p.intersects(p_) for p, p_ in zip(self.polygons, self.vor_polygons)]), 'All PhII polygons must intersect with corresponding Voronoi polygons'
    # Compute "matched" Voronoi polygons (i.e. bounded nicely)
    boundary = PlanarPolygon.from_image(self.ph2_mask)
    self.matched_display_vor_polygons = [p.intersection(boundary) for p in self.display_vor_polygons[:N]]
    self.matched_display_vor_polygons = [q for [q] in self.matched_display_vor_polygons] # Intersection must be simple and proper
    self.matched_vor_polygons = [p.set_res(*self.upp) for p in self.matched_display_vor_polygons]
    assert len(self.matched_vor_polygons) == N, 'Must have a matched Voronoi polygon for each compartment'

def fun(stage: str, spheroid: int, path: str):
    voxsize = get_voxel_size(f'{path}.czi', fmt='XY')
    mask = np.load(f'{path}.mask.matched.npy')
    boundary = PlanarPolygon.from_shape(mask[0].shape)
    
    cells = pickle.load(open(f'{path}.cell_polygons', 'rb')) # Use all cell polygons
    centroids = np.array([p.centroid() for p in cells])
    vors = boundary.voronoi_tessellate(centroids).polygons 
    assert len(vors) == len(cells), 'Must have a voronoi cell for each cell'

    cells_matched = pickle.load(open(f'{path}.cell_polygons.matched', 'rb'))
    centroids_matched = np.array([p.centroid() for p in cells_matched])
    dists = cdist(centroids_matched, centroids)
    idxs = np.argmin(dists, axis=1)
    assert np.unique(idxs).size == idxs.size, 'Must have a unique closest cell for each vor cell'
    assert np.allclose(0, dists.min(axis=1)), 'Must have a zero-distance cell for each vor cell'
    vors = np.array(vors)[idxs].tolist()
    assert len(vors) == len(cells_matched), 'Must have a voronoi cell for each cell'

    boundary = PlanarPolygon.from_image(mask[0])
    vors = [p.intersection(boundary) for p in vors]
    vors = [q for [q] in vors] # Intersection must be simple and proper
    vors = [p.rescale(voxsize) for p in vors]
    assert len(vors) == len(cells_matched), 'Must have a voronoi cell for each cell'
    pickle.dump(vors, open(f'{path}.voronoi.rescaled.matched', 'wb'))

# def fun(stage: str, spheroid: int, path: str):
#     cells = pickle.load(open(f'{path}.cell_polygons.rescaled', 'rb')) # Use all cell polygons
#     # boundary = pickle.load(open(f'{path}.boundary.rescaled', 'rb')) # Uses boundary of organism
#     compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled', 'rb'))
#     boundary = PlanarPolygon.from_chull_polygons(compartments) # Uses chull of compartments, more correct for taking Voronoi error
#     centroids = np.array([p.centroid() for p in cells])
#     centroids = centroids[boundary.contains(centroids)]
#     vors = boundary.voronoi_tessellate(centroids).polygons 
#     assert len(vors) == len(centroids), 'Must have a voronoi cell for each cell'
#     pickle.dump(vors, open(f'{path}.voronoi.rescaled', 'wb'))
#     # Match to cell-compartment pairs
#     matched_cells = pickle.load(open(f'{path}.cell_polygons.rescaled.matched', 'rb'))
#     matched_centroids = np.array([p.centroid() for p in matched_cells])
#     dists = cdist(matched_centroids, centroids)
#     idxs = np.argmin(dists, axis=1)
#     assert np.unique(idxs).size == idxs.size, 'Must have a unique closest cell for each vor cell'
#     assert np.allclose(0, dists.min(axis=1)), 'Must have a zero-distance cell for each vor cell'
#     vors = np.array(vors)[idxs].tolist()
#     assert len(vors) == len(matched_cells), 'Must have a voronoi cell for each cell'
#     pickle.dump(vors, open(f'{path}.voronoi.rescaled.matched', 'wb'))

if __name__ == '__main__':
    run_for_each_spheroid_topview(fun)