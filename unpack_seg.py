'''
Unpack .seg files into simpler data structures
'''

import pickle
import sys
from lib import *
from microseg.utils.data import get_voxel_size
import microseg
import microseg.data
import microseg.data.seg_2d
import matgeo
import matgeo.plane
import matgeo.ellipsoid
from matgeo import Ellipse
from ph2_segmentation import match_ph2_autofl

def unpack_seg(stage: str, spheroid: int, path: str):
    seg = pickle.load(open(f'{path}.seg', 'rb'))
    # Sanity check
    xyres = get_voxel_size(f'{path}.czi', fmt='XY')
    assert np.allclose(seg.upp, xyres)
    # Save images
    np.save(f'{path}.zproj.npy', seg.zproj)
    np.save(f'{path}.mask.npy', seg.mask)
    # Save boundary
    boundary_poly = seg.outline.migrate_OLD()
    boundary = Ellipse.from_poly(boundary_poly)
    pickle.dump(boundary, open(f'{path}.boundary', 'wb'))
    # Transpose
    pickle.dump(boundary.transpose(), open(f'{path}.boundary.transposed', 'wb'))
    # Rescale
    boundary = boundary.rescale(xyres)
    pickle.dump(boundary, open(f'{path}.boundary.rescaled', 'wb'))

def compare_extracted_polygons(stage: str, spheroid: int, path: str):
    seg = pickle.load(open(f'{path}.seg_ph2', 'rb'))
    compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
    print(f'Old compartments: {len(seg.orig_polygons)}, new compartments: {len(compartments)}')
    # # Try old mask procedure
    # mask = np.load(f'{path}.mask.npy')
    # ph2, autofl = match_ph2_autofl(mask[0], mask[1])
    # n_ph2, n_autofl = len(np.unique(ph2)) - 1, len(np.unique(autofl)) - 1
    # print(f'ph2 mask: {n_ph2}, autofl mask: {n_autofl}')

if __name__ == '__main__':
    sys.modules['seg_2d'] = microseg.data.seg_2d
    sys.modules['plane'] = matgeo.plane
    sys.modules['ellipsoid'] = matgeo.ellipsoid

    run_for_each_spheroid_topview(unpack_seg)
    # run_for_each_spheroid_topview(compare_extracted_polygons, log=False)