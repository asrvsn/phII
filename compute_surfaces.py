'''
Extract spheroid surfaces from voxel data
'''
from lib import *
import pyclesperanto_prototype as cle
import scipy
import scipy.ndimage
from scipy.spatial import ConvexHull
from microseg.utils.data import load_stack
from matgeo import Triangulation

# Parameters
TOPHAT_RADIUS = 5
BOUNDARY_DILATION = 1.05
SPOT_SIGMA = 0.5
OUTLINE_SIGMA = 0.5

def run():
    def fun(stage: str, spheroid: int, path: str):
        boundary = pickle.load(open(f'{path}.boundary', 'rb'))
        data = load_stack(f'{path}.czi')[:, :, :, 0].astype(np.float32) # ZYX
        voxsize = get_voxel_size(f'{path}.czi', fmt='XYZ')

        # 1. Equalize intensity
        mu = data.mean()
        for z in range(data.shape[0]):
            data[z] *= data[z].mean() / mu

        # 2. Subtract background
        data = cle.top_hat_box(data, radius_x=TOPHAT_RADIUS, radius_y=TOPHAT_RADIUS, radius_z=TOPHAT_RADIUS).get()

        # 3. Remove boundary
        mask = ~((boundary * BOUNDARY_DILATION).discretize(100)).to_mask(data.shape[1:])
        data[:, mask] = 0

        # 4. Extract 3D objects
        data = data.transpose(2, 1, 0) # ZYX -> XYZ
        mask = cle.voronoi_otsu_labeling(data, spot_sigma=SPOT_SIGMA, outline_sigma=OUTLINE_SIGMA).get()
        # vertices = scipy.ndimage.center_of_mass(
        #     np.ones_like(mask), labels=mask, index=np.arange(1, mask.max() + 1)
        # )
        vertices = np.argwhere(mask > 0).astype(np.float32) * voxsize

        # 5. Compute surface
        tri = Triangulation.from_convex_hull(vertices)
        pickle.dump(tri, open(f'{path}.surface', 'wb'))
        
    run_for_each_spheroid_topview(fun)

if __name__ == '__main__':
    run()