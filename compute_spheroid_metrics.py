'''
Compute metrics for each spheroid
'''

from lib import *
from scipy import stats
from typing import Optional
from dataclasses import dataclass

@dataclass
class SpheroidMetricContext:
    boundary: Ellipse
    ant_offspring: list[PlanarPolygon]

    def get_ap_axis(self) -> np.ndarray:
        '''
        Get anterior-posterior axis vector from centroids of gonidia polygons
        This is a basepointed vector with base at self.boundary.v (elliptical center)
        '''
        coms = np.array([p.centroid() for p in self.ant_offspring])
        line = Plane.fit_l2(coms) # Fit line thru gonidia centers of mass
        w = line.project_l2(self.boundary.v) # Closest point on line is orthogonal
        axis = w - self.boundary.v
        axis *= (axis @ self.boundary.M @ axis) ** (-0.5) # Rescale axis to hit ellipsoid
        return axis
    
@dataclass
class SpheroidMetric:
    fn: Callable[[SpheroidMetricContext], float]
    log: bool=False
    
spheroid_metrics = {
    'AP X': SpheroidMetric(
        fn = lambda ctx: ctx.get_ap_axis()[0],
    ),
    'AP Y': SpheroidMetric(
        fn = lambda ctx: ctx.get_ap_axis()[1],
    ),
    'Spheroid circular radius': SpheroidMetric(
        fn = lambda ctx: ctx.boundary.circular_radius(),
    ),
    'Spheroid aspect ratio': SpheroidMetric(
        fn = lambda ctx: ctx.boundary.aspect_ratio(),
    ),
}

@dataclass
class EnhancedSpheroidMetricContext(SpheroidMetricContext):
    cell_polygons: list[PlanarPolygon]
    compartment_polygons: list[PlanarPolygon]
    voronoi_polygons: list[PlanarPolygon]
    raw_metrics: pd.DataFrame
    nondimensionalized_metrics: pd.DataFrame
    normalized_metrics: pd.DataFrame

    @property
    def n_polygons(self) -> int:
        return len(self.cell_polygons)
    
    def get_hemisphere_index(self, hemisphere: str, polygons: list[PlanarPolygon]) -> np.ndarray:
        assert hemisphere in ['anterior', 'posterior']
        positions = np.array([p.centroid() for p in polygons]) - self.boundary.v
        lengths = np.linalg.norm(positions, axis=1)
        axis = self.get_ap_axis()
        signs = np.sign(positions @ axis)
        assert lengths.shape == signs.shape
        xvals = lengths * signs / (np.linalg.norm(axis) * 2)
        return (xvals >= 0) if hemisphere == 'anterior' else (xvals < 0)

enhanced_spheroid_metrics = {
    r'$k_{\rm gamma}(\alpha)$': SpheroidMetric(
        # k parameter of the gamma-distributed aspect ratios
        fn = lambda ctx:
            pt.fit_rv(ctx.raw_metrics['Aspect ratio'], stats.gamma, loc=1)[0][0],
        log = False
    ),
    r'$p_{\rm ks} (k_{\rm gamma}(\alpha))$': SpheroidMetric(
        # p-value of the gamma-distributed aspect ratios by the Kolmogorov-Smirnov statistic
        fn = lambda ctx:
            pt.fit_rv(ctx.raw_metrics['Aspect ratio'], stats.gamma, loc=1, pvalue=True, pvalue_statistic='ks')[1],
        log = False
    ),
    r'$k_{\rm gamma}(a_{\rm cz3})$': SpheroidMetric(
        # k parameter of the gamma-distributed areas
        fn = lambda ctx:
            pt.fit_rv(ctx.nondimensionalized_metrics['Compartment area'], stats.gamma)[0][0],
        log = True
    ),
    r'$p_{\rm ks} (k_{\rm gamma}(a_{\rm cz3}))$': SpheroidMetric(
        # p-value of the gamma-distributed areas by the Kolmogorov-Smirnov statistic
        fn = lambda ctx:
            pt.fit_rv(ctx.nondimensionalized_metrics['Compartment area'], stats.gamma, pvalue=True, pvalue_statistic='ks')[1],
        log = False
    ),
    r'$k_{\rm gamma}(a_{\rm cz3})$ (anterior)': SpheroidMetric(
        # k parameter of the gamma-distributed areas in anterior half
        fn = lambda ctx:
            pt.fit_rv(ctx.nondimensionalized_metrics['Compartment area'][ctx.get_hemisphere_index('anterior', ctx.compartment_polygons)], stats.gamma)[0][0],
        log = True
    ),
    r'$p_{\rm ks} (k_{\rm gamma}(a_{\rm cz3}))$ (anterior)': SpheroidMetric(
        # p-value of the gamma-distributed areas by the Kolmogorov-Smirnov statistic in anterior half
        fn = lambda ctx:
            pt.fit_rv(ctx.nondimensionalized_metrics['Compartment area'][ctx.get_hemisphere_index('anterior', ctx.compartment_polygons)], stats.gamma, pvalue=True, pvalue_statistic='ks')[1],
        log = False
    ),
    r'$k_{\rm gamma}(a_{\rm cz3})$ (posterior)': SpheroidMetric(
        # k parameter of the gamma-distributed areas in posterior half
        fn = lambda ctx:
            pt.fit_rv(ctx.nondimensionalized_metrics['Compartment area'][ctx.get_hemisphere_index('posterior', ctx.compartment_polygons)], stats.gamma)[0][0],
        log = True
    ),
    r'$p_{\rm ks} (k_{\rm gamma}(a_{\rm cz3}))$ (posterior)': SpheroidMetric(
        # p-value of the gamma-distributed areas by the Kolmogorov-Smirnov statistic in posterior half
        fn = lambda ctx:
            pt.fit_rv(ctx.nondimensionalized_metrics['Compartment area'][ctx.get_hemisphere_index('posterior', ctx.compartment_polygons)], stats.gamma, pvalue=True, pvalue_statistic='ks')[1],
        log = False
    ),
    r'Sum of second moments': SpheroidMetric(
        # Quantization energy (density) per unit measure
        # https://www.sciencedirect.com/science/article/pii/S0898122105001550
        # https://www.nature.com/articles/s41467-019-08360-5
        # E = n^(2/d) / (|Omega|^(1+2/d)) * sum_i E_i
        # d = 2 in our case
        fn = lambda ctx:
            np.sum([
                p.trace_M2(cp.centroid(), standardized=False) for p, cp in zip(ctx.compartment_polygons, ctx.cell_polygons)
            ]) * ctx.n_polygons / (np.array([p.area() for p in ctx.compartment_polygons]).sum() ** 2),
        log = False
    ),
    r'Median Voronoi error': SpheroidMetric(
        fn = lambda ctx:
            np.median([
                (1 - p.iou(v))
                for p, v in zip(ctx.compartment_polygons, ctx.voronoi_polygons)
            ]),
        log = False
    ),
}

all_spheroid_metrics = spheroid_metrics | enhanced_spheroid_metrics

def iterate_with_spheroid_context(fun: Callable[[str, int, str, SpheroidMetricContext], None], enhanced: bool=False, **kwargs):

    def outer_fun(stage: str, spheroid: int, path: str):
        boundary = pickle.load(open(f'{path}.boundary.rescaled', 'rb'))
        ant_offspring = pickle.load(open(f'{path}.offspring.anterior.rescaled', 'rb'))
        if enhanced:
            compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
            cells = pickle.load(open(f'{path}.cell_polygons.rescaled.matched', 'rb'))
            vors = pickle.load(open(f'{path}.voronoi.rescaled.matched', 'rb'))
            raw_metrics = pd.read_csv(f'{path}.polygon_metrics.raw.csv')
            nondimensionalized_metrics = pd.read_csv(f'{path}.polygon_metrics.nondimensionalized.csv')
            normalized_metrics = pd.read_csv(f'{path}.polygon_metrics.normalized.csv')
            ctx = EnhancedSpheroidMetricContext(
                boundary=boundary,
                ant_offspring=ant_offspring,
                cell_polygons=cells,
                compartment_polygons=compartments,
                voronoi_polygons=vors,
                raw_metrics=raw_metrics,
                nondimensionalized_metrics=nondimensionalized_metrics,
                normalized_metrics=normalized_metrics,
            )
        else:
            ctx = SpheroidMetricContext(
                boundary=boundary,
                ant_offspring=ant_offspring,
            )
        fun(stage, spheroid, path, ctx)

    run_for_each_spheroid_topview(outer_fun, **kwargs)

def run(enhanced: bool=False):
    columns = [
        'Stage',
        'Spheroid',
    ]
    columns += list(spheroid_metrics.keys())
    if enhanced:
        columns += list(enhanced_spheroid_metrics.keys())

    rows = []

    def fun(stage: str, spheroid: int, path: str, ctx: SpheroidMetricContext):
        '''
        Compute metrics for a given spheroid
        '''
        row = [stage, spheroid]
        for metric in spheroid_metrics.values():
            row.append(metric.fn(ctx))
        if enhanced:
            for metric in enhanced_spheroid_metrics.values():
                row.append(metric.fn(ctx))
        rows.append(row)

    iterate_with_spheroid_context(fun, enhanced=enhanced)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f'{DATA_PATH}/topview/all_spheroid_metrics.csv', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compute metrics for each spheroid")
    parser.add_argument('--enhanced', action='store_true', help='Enable enhanced mode')
    args = parser.parse_args()

    run(enhanced=args.enhanced)