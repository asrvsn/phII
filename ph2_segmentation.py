'''
PhII segmentation
'''
from typing import Tuple, List, Dict, Callable, Optional
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import cdist
import numpy as np
import numpy.linalg as la
import os
import pyqtgraph as pg
import asrvsn_mpl as pt
from microseg.data.seg_2d import Segmentation2D
from microseg.utils.mask import mask_to_com, mask_to_polygons, mask_to_adjacency
from microseg.utils.data import load_stack, get_voxel_size
from microseg.widgets.pg_gl import GrabbableGLViewWindow, GLZStackItem, GLTriangulationItem
import microseg.utils.pg as pgutil
from matgeo import Ellipse, PlanarPolygon, Plane, Sphere, Ellipsoid, Triangulation
from matgeo.voronoi import poly_bounded_voronoi
from im_utils import *
from asrvsn_math.array import flatten_list

def match_ph2_autofl(ph2_: np.ndarray, autofl_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Mask the PhII-autofl labels
    - Associates each PhII compartment with the closest autofluorescent cell, gives it the cell's label
    - Will re-label the PhII mask, but not the autofluorescence mask
    - Returns a strict 1-1 subset of both masks
    '''
    # Iterate through compartment labels, find containing somatic cells, and choose (if multiple) closest to COM
    cell_lbs_seen = set([0])
    cell_coms = mask_to_com(autofl_, as_dict=True, use_chull_if_invalid=True) # Cell centers of mass
    ph2, autofl = np.zeros_like(ph2_), np.zeros_like(autofl_)
    for ph2_lb in np.unique(ph2_)[1:]:
        ph2_idx = np.where(ph2_ == ph2_lb) # Indices of compartment label
        comp_com = np.mean(ph2_idx, axis=1) # Center of mass of compartment
        lbs, counts = np.unique(autofl_[ph2_idx], return_counts=True) # Somatic cell labels at position of compartment
        lbs, counts = lbs[lbs != 0], counts[lbs != 0] # Remove background label
        if lbs.size > 0:
            coms = np.array([cell_coms[lb] for lb in lbs]) # Centers of mass of somatic cells
            dists = np.linalg.norm(coms - comp_com, axis=1) # Distances between somatic cells and compartment com
            min_idx = np.argmin(dists) # Index of closest somatic cell
            lb = lbs[min_idx] # Label of closest somatic cell
            count = counts[min_idx] # Number of pixels in compartment with that label
            area_frac = count / np.sum(autofl_ == lb) # Fraction of somatic cell area within compartment
            if not (lb in cell_lbs_seen) and area_frac > 0.9: # If majority vote and not double-counted
                cell_lbs_seen.add(lb)
                autofl[autofl_ == lb] = lb
                ph2[ph2_idx] = lb
    assert np.allclose(np.unique(ph2), np.unique(autofl)), 'Ph2 and autofluorescence masks must have same labels'
    return ph2, autofl

class Metric:
    '''
    Scalar metric as a function of (compartment polygon, cell polygon, voronoi polygon)
    '''
    def __init__(self, 
            fn: Callable[[Tuple[PlanarPolygon,PlanarPolygon,PlanarPolygon]], float], 
            nondimensionalize: Callable[[np.ndarray], np.ndarray]=lambda xs: xs, # Nondimensionalize (standardize across instances)
            normalize: Callable[[np.ndarray], np.ndarray]=lambda xs: xs, # Normalize (affine-transform combined distribution for fitting)
        ):
        self.fn = fn
        self.nondimensionalize = nondimensionalize
        self.normalize = normalize

    def __call__(self, owner: 'Ph2Segmentation', ph2: PlanarPolygon, cell: PlanarPolygon, vor: PlanarPolygon) -> float:
        return self.fn(owner, ph2, cell, vor)

class Ph2Segmentation(Segmentation2D):
    '''
    PHII segmentation assuming:
    - channel 0 is PhII
    - channel 1 is autofluorescence
    - channel 2 is brightfield
    '''
    rdp_eps: float=0.0
    mask_dilate: int=1
    mask_erode: int=1
    ap_bins: int=10
    # Metrics computed per compartment-cell pair
    computed_metrics = {
        'Compartment area': Metric(
            lambda self, p, _, __: p.area(),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), 
        ),
        'Compartment perimeter': Metric(
            lambda self, p, _, __: p.perimeter(),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Compartment major axis': Metric(
            lambda self, p, _, __: 2 * p.stretches()[0] * 2,
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Major axis angle': Metric(
            lambda self, p, cp, _: angle_between_lines(p.major_axis(), self.ellipse.get_major_axis()) * 180 / np.pi,
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Compartment minor axis': Metric(
            lambda self, p, _, __: 2 * p.stretches()[1] * 2,
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Voronoi Area': Metric(
            lambda self, _, __, v: v.area(),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Cell area': Metric(
            lambda self, _, cp, __: cp.area(),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), 
        ),
        'Cell diameter': Metric(
            lambda self, _, cp, __: 2 * cp.circular_radius(),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Aspect ratio': Metric(
            lambda self, p, _, __: p.aspect_ratio(),
            normalize = lambda xs: (xs - 1) / (xs.mean() - 1), # Is nondimensional but not normalized
        ),
        'Cell aspect ratio': Metric(
            lambda self, _, cp, __: cp.aspect_ratio(),
            normalize = lambda xs: (xs - 1) / (xs.mean() - 1), # Is nondimensional but not normalized
        ),
        # Sensitive to affine transforms
        'Isoperimetric deficit': Metric(
            lambda self, p, _, __: p.isoperimetric_deficit(),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Circularity': Metric(
            lambda self, p, _, __: p.isoperimetric_ratio(),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        # Insensitive to affine transforms
        'ID (whitened)': Metric(
            lambda self, p, _, __: p.whiten().isoperimetric_deficit(),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Circularity (whitened)': Metric(
            lambda self, p, _, __: p.whiten().isoperimetric_ratio(),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Offset': Metric(
            lambda self, p, cp, _: la.norm(p.centroid() - cp.centroid()),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Squared offset': Metric(
            lambda self, p, cp, _: la.norm(p.centroid() - cp.centroid()) ** 2,
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Offset angle': Metric(
            lambda self, p, cp, _: angle_between_lines(cp.centroid() - p.centroid(), p.major_axis()) * 180 / np.pi,
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Offset (whitened)': Metric(
            lambda self, p, cp, _: p.mahalanobis_distance(cp.centroid()),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Circular radius': Metric(
            lambda self, p, _, __: p.circular_radius(),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Second moment': Metric(
            lambda self, p, cp, _: p.trace_M2(cp.centroid(), standardized=True), # Take the dimensionless form
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Voronoi error': Metric(
            lambda self, p, _, vp: (1 - p.iou(vp)), # Dimensionless
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),       
        'Voronoi neighbors': Metric(
            lambda self, _, __, vp: vp.n,
        ),
        'Cell AP position': Metric(
            lambda self, _, cp, __: self.get_ap_pos(cp.centroid()),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Compartment AP position': Metric(
            lambda self, p, _, __: self.get_ap_pos(p.centroid()),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Voronoi AP position': Metric(
            lambda self, _, __, vp: self.get_ap_pos(vp.centroid()),
            normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()), # Is nondimensional but not normalized
        ),
        'Distance to center': Metric(
            lambda self, p, _, __: la.norm(p.centroid() - self.ellipse.v),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Mahalanobis distance to center': Metric(
            lambda self, p, _, __: self.ellipse.mahalanobis_distance(p.centroid()),
            nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        ),
        'Chull-projected normal': Metric(
            lambda self, p, _, __: self.hull.project_poly_z(p).n @ np.array([0,0,-1]),
        ),
    }
    # Metrics computed for the entire segmentation
    computed_seg_metrics = {
        'AP radius':
            lambda self: 
                np.linalg.norm(self.get_ap_axis()),
        'Spheroid radius': 
            lambda self: 
                self.outline.circular_radius(),
        'Spheroid major axis':
            lambda self:
                self.outline.stretches()[0] * 2,
        'Spheroid minor axis':
            lambda self:
                self.outline.stretches()[1] * 2,
        'Spheroid aspect ratio':
            lambda self:
                self.outline.aspect_ratio(),
        r'$k_{\rm gamma}(\alpha)$':
            # k parameter of the gamma-distributed aspect ratios
            lambda self:
                pt.fit_rv(self.metrics['Aspect ratio']['raw'], stats.gamma, loc=1)[0][0],
        r'$p_{\rm ks} (k_{\rm gamma}(\alpha))$':
            # p-value of the gamma-distributed aspect ratios by the Kolmogorov-Smirnov statistic
            lambda self:
                pt.fit_rv(self.metrics['Aspect ratio']['raw'], stats.gamma, loc=1, pvalue=True, pvalue_statistic='ks')[1],
        r'$k_{\rm gamma}(a_{\rm cz3})$':
            # k parameter of the gamma-distributed areas
            lambda self:
                pt.fit_rv(self.metrics['Compartment area']['nondim'], stats.gamma)[0][0],
        r'$p_{\rm ks} (k_{\rm gamma}(a_{\rm cz3}))$':
            # p-value of the gamma-distributed areas by the Kolmogorov-Smirnov statistic
            lambda self:
                pt.fit_rv(self.metrics['Compartment area']['nondim'], stats.gamma, pvalue=True, pvalue_statistic='ks')[1],
        r'$k_{\rm gamma}(a_{\rm cz3})$ (anterior)':
            # k parameter of the gamma-distributed areas in anterior half
            lambda self:
                pt.fit_rv(self.metrics['Compartment area']['nondim'][self.get_hemisphere_index('anterior', 'Compartment AP position')], stats.gamma)[0][0],
        r'$p_{\rm ks} (k_{\rm gamma}(a_{\rm cz3}))$ (anterior)':
            # p-value of the gamma-distributed areas by the Kolmogorov-Smirnov statistic in anterior half
            lambda self:
                pt.fit_rv(self.metrics['Compartment area']['nondim'][self.get_hemisphere_index('anterior', 'Compartment AP position')], stats.gamma, pvalue=True, pvalue_statistic='ks')[1],
        r'$k_{\rm gamma}(a_{\rm cz3})$ (posterior)':
            # k parameter of the gamma-distributed areas in posterior half
            lambda self:
                pt.fit_rv(self.metrics['Compartment area']['nondim'][self.get_hemisphere_index('posterior', 'Compartment AP position')], stats.gamma)[0][0],
        r'$p_{\rm ks} (k_{\rm gamma}(a_{\rm cz3}))$ (posterior)':
            # p-value of the gamma-distributed areas by the Kolmogorov-Smirnov statistic in posterior half
            lambda self:
                pt.fit_rv(self.metrics['Compartment area']['nondim'][self.get_hemisphere_index('posterior', 'Compartment AP position')], stats.gamma, pvalue=True, pvalue_statistic='ks')[1],
        ## The following is equivalent to the Tessellation E_Q density only for monodisperse packings, which is not our case.
        ## See (2-4), https://www.nature.com/articles/s41467-019-08360-5 . As it is mentioned, the "dimensionless cellular energy" is in fact dependent upon its area.
        # r'Compartment $E_Q$':
        #     # Quantization energy of the compartment-cell pairs
        #     # https://journals.aps.org/pre/pdf/10.1103/PhysRevE.82.056109 
        #     lambda self:
        #         np.array([
        #             p.trace_M2(cp.centroid(), dimensionless=True) for p, cp in zip(self.polygons, self.cell_polygons)
        #         ]).mean(),
        #         # (2 * np.array([p.area() for p in self.polygons]).mean() ** 2),
        r'Sum of second moments':
            # Quantization energy (density) per unit measure
            # https://www.sciencedirect.com/science/article/pii/S0898122105001550
            # https://www.nature.com/articles/s41467-019-08360-5
            # E = n^(2/d) / (|Omega|^(1+2/d)) * sum_i E_i
            # d = 2 in our case
            lambda self:
                np.sum([
                    p.trace_M2(cp.centroid(), standardized=False) for p, cp in zip(self.polygons, self.cell_polygons)
                ]) * self.n_polygons / (self.total_polygon_area ** 2),
        # r'Compartment $E^2$':
        #     # E2 energy of the compartment-cell pairs
        #     # https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.L042001
        #     lambda self:
        #         np.array([
        #             p.E2_energy(cp.centroid()) for p, cp in zip(self.polygons, self.cell_polygons)
        #         ]).mean(),
        # r'Voronoi $E_Q$': 
        #     # Quantization energy of the containing region by the cells (only valid if segmented compartments actually cover the entire region)
        #     lambda self: 
        #         self.outline.voronoi_quantization_energy(self.cell_coms, dimensionless=True),
        r'Median Voronoi error':
            lambda self:
                np.median([
                    (1 - p.iou(v))
                    for p, v in zip(self.polygons, self.matched_vor_polygons)
                ]),
        'Packing fraction':
            lambda self:
                self.total_polygon_area / np.array([v.area() for v in self.matched_vor_polygons]).sum(),
        'Mean juvenile radius':
            lambda self:
                np.mean([p.circular_radius() for p in self.all_offspring]),
    }
    
    def __init__(self, 
            seg: Segmentation2D, 
            anterior_offspring: List[PlanarPolygon], 
            all_offspring: List[List[PlanarPolygon]],
            pathname: str,
            img_path: str,
            recompute: bool=False,
        ):
        self.pathname = pathname
        self.img_path = img_path
        self.czxy = seg.czxy
        self.upp = seg.upp
        self.settings = seg.settings
        self.zproj = seg.zproj
        self.img = seg.img
        self.outline = seg.outline
        self.cp_mask = seg.cp_mask
        self.mask = seg.mask
        assert not self.outline is None, 'Must provide outline'
        self.display_outline = self.outline.copy()
        self.outline = self.outline.set_res(*self.upp) # Do not recompute
        # Data
        self.ph2 = self.zproj[0]
        self.autofl = self.zproj[1]
        self.brightfield = self.zproj[2]
        self.combined = combine_grayscale(self.ph2, self.autofl)
        self.combined_rgb = combine_rgb(self.ph2, self.autofl)
        # Masks
        self.ph2_mask, self.autofl_mask = match_ph2_autofl(self.mask[0], self.mask[1])
        print(f'Matched masks')
        # Offspring
        self.display_anterior_offspring = [p.hullify() for p in anterior_offspring]
        self.anterior_offspring = [p.set_res(*self.upp) for p in self.display_anterior_offspring]
        self.display_all_offspring = all_offspring # Retain z-structure for display at correct focal plane
        self.all_offspring = [p.set_res(*self.upp) for p in flatten_list(all_offspring)]
        print(f'Loaded offspring')
        # # Adjacency
        # self.adjacency = mask_to_adjacency(self.ph2_mask)
        # # Polygons & metrics
        # self.compute_polygons()
        # self.compute_metrics()

    def recompute(self):
        self.compute_polygons()
        self.compute_triangulations()
        self.compute_metrics()
        
    def compute_polygons(self):
        '''
        Compute the polygons
        '''
        # Polygons
        self.display_polygons = [PlanarPolygon(p) for p in mask_to_polygons(self.ph2_mask, rdp_eps=self.rdp_eps, erode=self.mask_erode, dilate=self.mask_dilate)]
        self.display_cell_polygons = [PlanarPolygon(p) for p in mask_to_polygons(self.autofl_mask, rdp_eps=self.rdp_eps, erode=self.mask_erode, dilate=self.mask_dilate)]
        self.display_cell_coms = np.array([p.centroid() for p in self.display_cell_polygons])
        self.display_ellipse = Ellipse.from_poly(self.display_outline)
        ## Rescale to proper units
        self.polygons = [p.set_res(*self.upp) for p in self.display_polygons]
        self.cell_polygons = [p.set_res(*self.upp) for p in self.display_cell_polygons]
        self.cell_coms = np.array([p.centroid() for p in self.cell_polygons])
        self.ellipse = Ellipse.from_poly(self.outline)
        ## Compute Voronoi
        self.compute_voronoi_polygons()
        # print('Computed polygons and ellipse')

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
        # print('Computed Voronoi polygons')

    def compute_triangulations(self):
        self.stack = load_stack(self.img_path)[:, :, :, 0].transpose(1,2,0) # img is ZXYC
        scale = get_voxel_size(self.img_path, fmt='XYZ')
        # Compute XY coordinates of stack
        x = np.arange(self.stack.shape[0]) 
        y = np.arange(self.stack.shape[1]) 
        xy = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1).reshape(-1,2)
        # Get mask of those contained inside the bounding ellipse
        mask = self.ellipse.contains(xy * scale[:2])
        xy_invalid = xy[~mask]
        # Set rest to zero
        self.stack[xy_invalid[:,0], xy_invalid[:,1], :] = 0
        # Compute tri and hull
        zmax = self.stack.shape[2] * scale[2]
        self.tri = Triangulation.from_volume(self.stack, method='marching_cubes', spacing=scale) + np.array([0,0,zmax * 0.5])
        self.hull = self.tri.hullify()
        
    def compute_metrics(self):
        '''
        Compute the polygon-derived metrics
        '''
        self.ap_axis = None # Anterior-posterior axis
        self.metrics = dict() # Metrics
        for name, metric in Ph2Segmentation.computed_metrics.items():
            xs = np.array([metric(self, *args) for args in zip(self.polygons, self.cell_polygons, self.matched_vor_polygons)]) # Computed using correctly scaled polygons
            ys = metric.nondimensionalize(xs.copy())
            zs = metric.normalize(ys.copy())
            self.metrics[name] = {
                'raw': xs,
                'nondim': ys,
                'normal': zs,
            }
        self.seg_metrics = dict()
        for name, metric in Ph2Segmentation.computed_seg_metrics.items():
            self.seg_metrics[name] = metric(self)
        # print('Computed metrics')
    
    def get_ap_axis(self) -> np.ndarray:
        '''
        Get anterior-posterior axis vector from centroids of gonidia polygons
        '''
        if self.ap_axis is None:
            coms = np.array([p.centroid() for p in self.anterior_offspring])
            line = Plane.fit_l2(coms) # Fit line thru gonidia centers of mass
            w = line.project_l2(self.ellipse.v) # Closest point on line is orthogonal
            axis = w - self.ellipse.v
            axis *= (axis @ self.ellipse.M @ axis) ** (-0.5) # Rescale axis to hit ellipsoid
            self.ap_axis = axis
        return self.ap_axis
    
    def get_ap_axis_OLD(self) -> np.ndarray:
        '''
        Get ellipsoid with stretches oriented along A-P axis (major stretch towards anterior)
        '''
        ell = self.ellipse.copy()
        axis = ell.get_major_axis()
        one_side = ((self.cell_coms - ell.v) @ axis) >= 0 # True if on one side of the ellipsoid
        polys = np.array(self.polygons)
        one_side_mean_area = np.mean([p.area() for p in polys[one_side]])
        other_side_mean_area = np.mean([p.area() for p in polys[~one_side]])
        side_is_anterior = one_side_mean_area > other_side_mean_area # True if anterior side is larger
        if not side_is_anterior:
            axis *= -1
        return axis

    def get_ap_pos(self, pos: np.ndarray) -> np.ndarray:
        '''
        Get positions along AP axis in 2D
        '''
        v, w = self.ellipse.v, self.ellipse.v + self.get_ap_axis()
        line = Plane.fit_l2(np.array([v, w]))
        pos = line.project_l2(pos)
        return pos
    
    def get_hemisphere_index(self, ap: str, posm: str) -> np.ndarray:
        '''
        Get indices corresponding to given metric in hemisphere
        '''
        assert ap in ['anterior', 'posterior']
        positions = self.metrics[posm]['raw'].copy() - self.ellipse.v
        lengths = la.norm(positions, axis=1)
        signs = np.sign(positions @ self.get_ap_axis())
        assert lengths.shape == signs.shape
        xvals = lengths * signs / (self.seg_metrics['AP radius'] * 2) # Convert 2D positions to percentages along AP axis
        return (xvals >= 0) if ap == 'anterior' else (xvals < 0)

    def plot_ap(self, ax: plt.Axes, linewidth=1, mode='radius', gonidia=True, pts=None, labels=False, label_color='black', fontsize=22):
        pt.ax_ellipse(ax, self.display_ellipse, color='white', axes=False, linewidth=linewidth)
        v = self.display_ellipse.v 
        # gonidia = [Sphere.from_poly(p).transpose(1, 0) for p in self.display_all_offspring] # Not all at same z generally
        coms = np.array([p.centroid() for p in self.display_anterior_offspring])
        axis = self.get_ap_axis() / np.array(self.upp)
        if gonidia:
            # Plot gonidia 
            # pt.ax_ellipses(ax, gonidia, color='lightgreen', axes=False, linewidth=linewidth)
            ax.scatter(coms[:,0], coms[:,1], marker='o', s=100, color='lightgreen', zorder=10)
            # Plot line thru gonidia
            gon_coms = np.array([p.centroid() for p in self.anterior_offspring])
            gon_line = Plane.fit_l2(gon_coms) # Fit line thru gonidia centers of mass
            gon_pts = self.ellipse.plane_intersection(gon_line)
            assert gon_pts.shape[0] == 2
            gon_pts /= np.array(self.upp)
            ax.plot(gon_pts[:,0], gon_pts[:,1], color='lightgreen', zorder=10, linewidth=linewidth)
        w = v + axis
        if mode == 'radius':
            ax.scatter([v[0]], [v[1]], marker='o', s=50, color='lightsalmon', zorder=10)
            ax.plot([v[0], w[0]], [v[1], w[1]], color='yellow', zorder=10, linewidth=linewidth)
        elif mode == 'diameter':
            u = v - axis
            ax.plot([u[0], w[0]], [u[1], w[1]], color='yellow', zorder=10, linewidth=linewidth)
        else:
            raise NotImplementedError
        if not pts is None:
            pts_ = self.get_ap_pos(pts)
            pts, pts_ = pts / np.array(self.upp), pts_ / np.array(self.upp)
            for p1, p2 in zip(pts, pts_):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightblue', zorder=10, linewidth=1)
            ax.scatter(pts[:,0], pts[:,1], marker='o', s=10, color='lightblue', zorder=10)
            ax.scatter(pts_[:,0], pts_[:,1], marker='x', s=10, color='magenta', zorder=10)
        if labels:
            for lbl, sign in [('A', 1), ('P', -1)]:
                v_ = v + axis * 1.05 * sign
                ax.text(v_[0], v_[1], r'\textbf{'+lbl+r'}', fontsize=fontsize, ha='center', va='center', color=label_color)
            

    def plot_ap_OLD(self, ax: plt.Axes, text_color='white', **kwargs):
        pt.ax_ellipse(ax, self.display_ellipse, **kwargs)
        _, rs = self.display_ellipse.get_axes_stretches()
        axis = self.get_ap_axis_OLD()
        e1 = self.display_ellipse.v + axis * rs[0] * 1.05
        e2 = self.display_ellipse.v - axis * rs[0] * 1.05
        # ax.text(e1[0], e1[1], r'\textbf{A}', fontsize=22, ha='center', va='center', color=text_color)
        # ax.text(e2[0], e2[1], r'\textbf{P}', fontsize=22, ha='center', va='center', color=text_color)
        
    def export_figure(self, path: str):
        '''
        Export segmentation figure
        '''
        print(f'Exporting segmentation figure to {path}...')
        # Original curated segmentations
        pt.plot_ax((0,0), 'Ph2 (original)', lambda ax: pt.ax_im_mask(ax, self.ph2, self.mask[0]))
        pt.plot_ax((1,0), 'Autofl (original)', lambda ax: pt.ax_im_mask(ax, self.autofl, self.mask[1]))
        # Matched segmentations
        pt.plot_ax((0,1), 'Ph2 (matched)', lambda ax: pt.ax_im_mask(ax, self.ph2, self.ph2_mask))
        pt.plot_ax((1,1), 'Autofl (matched)', lambda ax: pt.ax_im_mask(ax, self.autofl, self.autofl_mask))
        # Polygons
        pt.plot_ax((0,2), 'Matched polygons', lambda ax: pt.ax_im_gray(ax, self.ph2))
        pt.modify_ax((0,2), lambda ax: pt.ax_planar_polygons(ax, self.display_polygons))
        pt.modify_ax((0,2), lambda ax: pt.ax_planar_polygons(ax, self.display_cell_polygons))
        # Outline
        pt.plot_ax((2,0), 'Outline', lambda ax: pt.ax_im_gray(ax, self.brightfield))
        pt.modify_ax((2,0), lambda ax: pt.ax_planar_polygons(ax, [self.display_outline]))
        pt.modify_ax((2,0), lambda ax: pt.ax_ellipse(ax, self.display_ellipse))
        # Real scale
        pt.plot_ax((2,1), 'Real scale', lambda ax: pt.ax_planar_polygons(ax, [self.outline]))
        pt.modify_ax((2,1), lambda ax: pt.ax_ellipse(ax, self.ellipse))
        pt.modify_ax((2,1), lambda ax: pt.ax_planar_polygons(ax, self.polygons))
        pt.modify_ax((2,1), lambda ax: pt.ax_planar_polygons(ax, self.cell_polygons))
        pt.modify_ax((2,1), lambda ax: ax.invert_yaxis()) # Consistent with images
        # Anterior-posterior axis
        pt.plot_ax((2,2), 'Anterior-posterior axis', lambda ax: pt.ax_planar_polygons(ax, [self.outline]))
        pt.modify_ax((2,2), lambda ax: pt.ax_ellipse(ax, self.ellipse))
        pt.modify_ax((2,2), lambda ax: pt.ax_planar_polygons(ax, self.polygons, cmap=lambda _: pt.map_colors(self.ap_xvalues, 'continuous')))
        pt.modify_ax((2,2), lambda ax: ax.invert_yaxis()) # Consistent with images
        # Voronoi
        pt.plot_ax((0,3), 'Voronoi', lambda ax: pt.ax_im_gray(ax, self.img[0]))
        pt.modify_ax((0,3), lambda ax: pt.ax_planar_polygons(ax, self.display_vor_polygons))
        pt.modify_ax((0,3), lambda ax: pt.ax_planar_polygons(ax, self.all_display_cell_polygons))
        pt.modify_ax((0,3), lambda ax: ax.scatter(self.all_display_cell_coms[:, 0], self.all_display_cell_coms[:, 1], c='white', s=5))
        # Compartments + Voronoi neighbors
        pt.plot_ax((0,4), 'Ph2+Voronoi+neighbors', lambda ax: pt.ax_im_mask(ax, self.ph2, self.mask[0]))
        pt.modify_ax((0,4), lambda ax: pt.ax_planar_polygons(ax, self.display_vor_polygons, alpha=0., linewidth=0.5, linecolor='white'))
        pt.modify_ax((0,4), lambda ax: ax.scatter(self.all_cell_coms[:, 0], self.all_cell_coms[:, 1], c='white', s=2))
        pt.modify_ax((0,4), lambda ax: pt.ax_planar_polygons(ax, self.matched_display_vor_polygons, alpha=0., linewidth=0.5, linecolor='orange'))
        pt.modify_ax((0,4), lambda ax: ax.scatter(self.cell_coms[:, 0], self.cell_coms[:, 1], c='orange', s=2))
        pt.modify_ax((0,4), lambda ax: pt.ax_labels(ax, self.cell_coms, [v.n for v in self.matched_display_vor_polygons], fontsize=8, color='white'))
        # Voronoi (matched)
        pt.plot_ax((1,2), 'Voronoi (matched)', lambda ax: pt.ax_im_gray(ax, self.img[0]))
        pt.modify_ax((1,2), lambda ax: pt.ax_planar_polygons(ax, self.matched_display_vor_polygons))
        pt.modify_ax((1,2), lambda ax: ax.scatter(self.display_cell_coms[:, 0], self.display_cell_coms[:, 1], c='white', s=5))
        # Voronoi overlaied
        pt.plot_ax((1,3), 'Voronoi + PhII', lambda ax: pt.ax_im_gray(ax, self.ph2))
        pt.modify_ax((1,3), lambda ax: pt.ax_planar_polygons(ax, self.display_polygons, labels=True))
        pt.modify_ax((1,3), lambda ax: pt.ax_planar_polygons(ax, self.matched_display_vor_polygons, labels=True))
        # Outline, ellipse, AP axis
        pt.plot_ax((1,4), 'Outline/Ellipse', lambda ax: pt.ax_planar_polygons(ax, [self.outline]))
        pt.modify_ax((1,4), lambda ax: pt.ax_ellipse(ax, self.ellipse))
        pt.modify_ax((1,4), lambda ax: ax.scatter(self.all_cell_coms[:, 0], self.all_cell_coms[:, 1], c='black', s=2))
        # Voronoi error
        v_err = flatten_list([p.symmetric_difference(v) for p, v in zip(self.display_polygons, self.matched_display_vor_polygons)])
        pt.plot_ax((2,3), 'Voronoi error', lambda ax: pt.ax_im_gray(ax, self.ph2))
        pt.modify_ax((2,3), lambda ax: pt.ax_planar_polygons(ax, v_err))
        v_err_real = flatten_list([p.symmetric_difference(v) for p, v in zip(self.polygons, self.matched_vor_polygons)])
        pt.plot_ax((2,4), 'Voronoi error (real scale)', lambda ax: pt.ax_planar_polygons(ax, v_err_real))
        # Save
        pt.save_plots(path)

    def get_raw_slice(self, i: int) -> np.ndarray:
        return load_stack(self.img_path)[i] # img is ZXYC
    
    def project_to_ellipsoid(self, ell: Ellipsoid, poly: PlanarPolygon, cell_poly: PlanarPolygon) -> Tuple[PlanarPolygon, PlanarPolygon]:
        poly = ell.project_poly_z(poly)
        cell_poly = ell.project_poly_z(cell_poly, plane=poly.plane) # Use same plane as poly
        return poly, cell_poly

    def render_ellipsoid_projection(self):
        ell = self.ellipse.revolve_major(v_z=-1.2*self.ellipse.get_minor_radius())
        polys = [p.embed_XY() for p in self.polygons]
        cell_polys = [p.embed_XY() for p in self.cell_polygons]
        proj_all_polys = np.array([self.project_to_ellipsoid(ell, p, c) for p, c in zip(self.polygons, self.cell_polygons)])
        centers = Plane.XY().reverse_embed(np.array([p.centroid() for p in self.cell_polygons])) # Use cells as centers
        proj_polys = proj_all_polys[:,0]
        proj_cell_polys = proj_all_polys[:,1]
        proj_centers = np.array([p.plane.reverse_embed(p.centroid()) for p in proj_cell_polys]) # Use cells as centers
        def fun(vw):
            center = centers.mean(axis=0)
            viewsize = la.norm(centers - proj_centers, axis=1).max()
            vw.opts['center'] = pg.Vector(*(center))
            vw.setCameraPosition(distance=viewsize * 1.3)
            pgutil.ppolygons_3d(vw, polys, centers)
            # pgutil.ppolygons_3d(vw, cell_polys, centers)
            pgutil.ppolygons_3d(vw, proj_polys, proj_centers)
            # pgutil.ppolygons_3d(vw, proj_cell_polys, proj_centers)
            pgutil.ellipsoid_3d(vw, ell)
        return pgutil.run_gl(fun)
    
    def render_chull_projection(self):
        polys = [p.embed_XY() for p in self.polygons]
        poly_centers = Plane.XY().reverse_embed(np.array([p.centroid() for p in self.polygons]))
        proj_polys = [self.hull.project_poly_z(p) for p in self.polygons]
        proj_poly_centers = np.array([p.plane.reverse_embed(p.centroid()) for p in proj_polys])
        def fun(vw):
            center = self.tri.pts.mean(axis=0)
            vw.opts['center'] = pg.Vector(*(center))
            viewsize = la.norm(self.tri.pts - center, axis=1).max()
            vw.setCameraPosition(distance=viewsize * 1.3)
            # vol = GLZStackItem(self.czxy[0], xyz_scale=self.upp)
            surf = GLTriangulationItem(self.tri)
            vw.addItem(surf)
            pgutil.ppolygons_3d(vw, polys, poly_centers)
            pgutil.ppolygons_3d(vw, proj_polys, proj_poly_centers)
        return pgutil.run_gl(fun)

    @property
    def n_polygons(self) -> int:
        return len(self.polygons)
    
    @property
    def total_polygon_area(self) -> float:
        return np.sum([p.area() for p in self.polygons])
    

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (la.norm(v1) * la.norm(v2))

def angle_between_lines(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.arccos(abs(cosine_similarity(v1, v2)))
