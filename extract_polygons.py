'''
Extract polygons from spheroid segmentation masks
'''

from lib import *
from matgeo.plane import PlanarPolygonPacking
from microseg.utils.data import get_voxel_size
from ph2_segmentation import match_ph2_autofl

# Parameters
ERODE = 1
DILATE = 1
# METHOD = 'marching_squares'
METHOD = 'standard'
BOUNDARY_DISCRETIZATION = 100


def extract_polygons(stage: str, spheroid: int, path: str):
    mask = np.load(f'{path}.mask.npy')
    boundary = pickle.load(open(f'{path}.boundary', 'rb')).discretize(BOUNDARY_DISCRETIZATION)
    # Extract from mask
    compartment_polygons = PlanarPolygonPacking.from_mask(mask[0], erode=ERODE, dilate=DILATE, method=METHOD, boundary=boundary)
    pickle.dump(compartment_polygons.polygons, open(f'{path}.compartment_polygons', 'wb'))
    cell_polygons = PlanarPolygonPacking.from_mask(mask[1], erode=ERODE, dilate=DILATE, method=METHOD, boundary=boundary)
    pickle.dump(cell_polygons.polygons, open(f'{path}.cell_polygons', 'wb'))
    # Rescale
    voxsize = get_voxel_size(f'{path}.czi', fmt='XY')
    compartment_polygons = compartment_polygons.rescale(voxsize)
    ## Sanity check
    boundary = pickle.load(open(f'{path}.boundary.rescaled', 'rb')).discretize(BOUNDARY_DISCRETIZATION)
    assert np.isclose(compartment_polygons.surface.area(), boundary.area())
    pickle.dump(compartment_polygons.polygons, open(f'{path}.compartment_polygons.rescaled', 'wb'))
    cell_polygons = cell_polygons.rescale(voxsize)
    assert np.isclose(cell_polygons.surface.area(), boundary.area())
    pickle.dump(cell_polygons.polygons, open(f'{path}.cell_polygons.rescaled', 'wb'))
    # Match cell-compartment pairs
    ph2, autofl = match_ph2_autofl(mask[0], mask[1])
    mask_matched = np.stack([ph2, autofl], axis=0)
    assert mask_matched.shape == (2, mask.shape[1], mask.shape[2]), 'Must have a matched mask'
    np.save(f'{path}.mask.matched.npy', mask_matched)
    # compartment_polygons, cell_polygons = PlanarPolygonPacking.match_by_containment(compartment_polygons, cell_polygons)
    compartment_polygons = PlanarPolygonPacking.from_mask(ph2, erode=ERODE, dilate=DILATE, method=METHOD, boundary=boundary)
    pickle.dump(compartment_polygons.polygons, open(f'{path}.compartment_polygons.matched', 'wb'))
    cell_polygons = PlanarPolygonPacking.from_mask(autofl, erode=ERODE, dilate=DILATE, method=METHOD, boundary=boundary)
    pickle.dump(cell_polygons.polygons, open(f'{path}.cell_polygons.matched', 'wb'))
    assert len(compartment_polygons) == len(cell_polygons)
    # Rescale
    compartment_polygons = compartment_polygons.rescale(voxsize)
    pickle.dump(compartment_polygons.polygons, open(f'{path}.compartment_polygons.rescaled.matched', 'wb'))
    cell_polygons = cell_polygons.rescale(voxsize)
    pickle.dump(cell_polygons.polygons, open(f'{path}.cell_polygons.rescaled.matched', 'wb'))

if __name__ == '__main__':
    run_for_each_spheroid_topview(extract_polygons)