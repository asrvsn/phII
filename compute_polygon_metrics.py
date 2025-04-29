'''
Compute metrics for cell-compartment pairs
'''

import numpy.linalg as la
from asrvsn_math.vectors import angle_between_lines
from scipy import stats
from itertools import repeat
from dataclasses import dataclass

from lib import *

# Defined metrics and their metadata

@dataclass
class MetricContext:
    cell: PlanarPolygon
    compartment: PlanarPolygon
    voronoi: PlanarPolygon
    boundary: PlanarPolygon
    ap_vec: np.ndarray

    def get_ap_value(self, pos: np.ndarray) -> float:
        u, w = self.boundary.v - self.ap_vec, self.boundary.v + self.ap_vec
        line = Plane.fit_l2(np.array([u, w]))
        val = la.norm(line.project_l2(pos) - u) / la.norm(w - u)
        return val

class Metric:
    '''
    Scalar metric as a function of (compartment polygon, cell polygon, voronoi polygon)
    '''
    def __init__(self, 
            fn: Callable[[MetricContext], float], 
            nondimensionalize: Callable[[np.ndarray], np.ndarray]=lambda xs: xs, # Nondimensionalize (standardize across instances)
            normalize: Callable[[np.ndarray], np.ndarray]=lambda xs: xs, # Normalize (affine-transform combined distribution for fitting)
            units: str=None,
            limits: Tuple[float,float]=(-np.inf, np.inf),
            label: str=None,
            label_normal: str=None,
            rvs: list=[],
            corr_transform: callable=lambda xs: xs,
            corr_transform_label: str=None,
            floc: float=None,
        ):
        self.fn = fn
        self.nondimensionalize = nondimensionalize
        self.normalize = normalize
        self.units = units
        self.limits = limits
        self.label = label
        self.label_normal = label_normal
        self.rvs = rvs
        self.corr_transform = corr_transform
        self.corr_transform_label = corr_transform_label
        self.floc = floc

    def __call__(self, ctx: MetricContext) -> float:
        return self.fn(ctx)

all_metrics = {
    'Compartment AP value': Metric(
        lambda ctx: ctx.get_ap_value(ctx.compartment.centroid()),
        label = r'$p_{cz3}$',
        units = 'unitless',
    ),
    'Cell AP value': Metric(
        lambda ctx: ctx.get_ap_value(ctx.cell.centroid()),
        label = r'$p_{cell}$',
        units = 'unitless',
    ),
    'Voronoi AP value': Metric(
        lambda ctx: ctx.get_ap_value(ctx.voronoi.centroid()),
        label = r'$p_{vor}$',
        units = 'unitless',
    ),
    'Compartment area': Metric(
        lambda ctx: ctx.compartment.area(),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$a_{cz3}$',
        label_normal = r'$(a_{cz3}-\min(a_{cz3}))/(\overline{a_{cz3}}-\min(a_{cz3}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(a_{cz3})$',
        rvs = [stats.gamma],
        limits = (0, np.inf),
        units = r'$\mu m^2$'
    ),
    'Compartment perimeter': Metric(
        lambda ctx: ctx.compartment.perimeter(),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$p_{cz3}$',
        label_normal = r'$(p_{cz3}-\min(p_{cz3}))/(\overline{p_{cz3}}-\min(p_{cz3}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(p_{cz3})$',
        rvs = [stats.gamma],
        limits = (0, np.inf),
        units = r'$\mu m$'
    ),
    'Compartment major axis': Metric(
        lambda ctx: 2 * ctx.compartment.stretches()[0] * 2,
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$d_{\text{maj}}$',
        label_normal = r'$(d_{\text{maj}}-\min(d_{\text{maj}}))/(\overline{d_{\text{maj}}}-\min(d_{\text{maj}}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(d_{\text{maj}})$',
        rvs = [stats.gamma],
        limits = (0, np.inf),
        units = r'$\mu m$'
    ),
    'Major axis angle': Metric(
        lambda ctx: angle_between_lines(ctx.compartment.major_axis(), ctx.boundary.get_major_axis()) * 180 / np.pi,
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$\theta_{\rm cz3}$',
        label_normal = r'$(\theta_{\rm cz3}-\min(\theta_{\rm cz3}))/(\overline{\theta_{\rm cz3}}-\min(\theta_{\rm cz3}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(\theta_{\rm cz3})$',
        rvs = [stats.gamma],
        limits = (0, 90),
        units = r'degrees'
    ),
    'Compartment minor axis': Metric(
        lambda ctx: 2 * ctx.compartment.stretches()[1] * 2,
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$d_{\text{min}}$',
        label_normal = r'$(d_{\text{min}}-\min(d_{\text{min}}))/(\overline{d_{\text{min}}}-\min(d_{\text{min}}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(d_{\text{min}})$',
        rvs = [stats.gamma],
        limits = (0, np.inf),
        units = r'$\mu m$'
    ),
    'Voronoi Area': Metric(
        lambda ctx: ctx.voronoi.area(),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$a_v$',
        label_normal = r'$(a_v-\min(a_v))/(\overline{a_v}-\min(a_v))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(a_v)$',
        rvs = [stats.gamma],
        limits = (0, np.inf),
        units = r'$\mu m^2$'
    ),
    'Cell area': Metric(
        lambda ctx: ctx.cell.area(),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$a_{cell}$',
        label_normal = r'$(a_{cell}-\min(a_{cell}))/(\overline{a_{cell}}-\min(a_{cell}))$',
        corr_transform = lambda x: x,
        corr_transform_label = r'$a_{cell}$',
        rvs = [],
        limits = (0, 80),
        units = r'$\mu m^2$'
    ),
    'Cell diameter': Metric(
        lambda ctx: 2 * ctx.cell.circular_radius(),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$d_{cell}$',
        label_normal = r'$(d_{cell}-\min(d_{cell}))/(\overline{d_{cell}}-\min(d_{cell}))$',
        corr_transform = lambda x: x,
        corr_transform_label = r'$d_{cell}$',
        rvs = [],
        limits = (0, np.inf),
        units = r'$\mu m$'
    ),
    'Aspect ratio': Metric(
        lambda ctx: ctx.compartment.aspect_ratio(),
        normalize = lambda xs: (xs - 1) / (xs.mean() - 1),
        label = r'$\alpha$',
        label_normal = r'$(\alpha-\min(\alpha))/(\overline{\alpha}-\min(\alpha))$',
        corr_transform = lambda x: np.log10(x-1),
        corr_transform_label = r'$\log_{10}(\alpha-1)$',
        rvs = [stats.gamma],
        limits = (0, 2),
        units = 'unitless',
        floc = 1.
    ),
    'Cell aspect ratio': Metric(
        lambda ctx: ctx.cell.aspect_ratio(),
        normalize = lambda xs: (xs - 1) / (xs.mean() - 1),
        label = r'$\alpha_{\rm cell}$',
        label_normal = r'$(\alpha_{\rm cell}-\min(\alpha_{\rm cell}))/(\overline{\alpha_{\rm cell}}-\min(\alpha_{\rm cell}))$',
        corr_transform = lambda x: np.log10(x-1),
        corr_transform_label = r'$\log_{10}(\alpha_{\rm cell}-1)$'
    ),
    'Isoperimetric deficit': Metric(
        lambda ctx: ctx.compartment.isoperimetric_deficit(),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$\delta$',
        label_normal = r'$(\delta-\min(\delta))/(\overline{\delta}-\min(\delta))$',
        corr_transform = lambda x: np.log10(x - PlanarPolygon.id_ngon(6)),
        corr_transform_label = r'$\log_{10}(\delta - \delta_6)$',
        rvs = [],
        limits = (0, 0.35),
        units = 'unitless'
    ),
    'Circularity': Metric(
        lambda ctx: ctx.compartment.isoperimetric_ratio(),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$q$',
        label_normal = r'$(q-\min(q))/(\overline{q}-\min(q))$',
        corr_transform = lambda x: x / PlanarPolygon.ir_ngon(6),
        corr_transform_label = r'$q / q_{\text{hex}}$',
        rvs = [],
        limits = (0, 1),
        units = 'unitless'
    ),
    'ID (whitened)': Metric(
        lambda ctx: ctx.compartment.whiten().isoperimetric_deficit(),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$\tilde{\delta}$',
        label_normal = r'$(\tilde{\delta}-\min(\tilde{\delta}))/(\overline{\tilde{\delta}}-\min(\tilde{\delta}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(\tilde{\delta})$',
        rvs = [],
        limits = (0, np.inf),
        units = 'unitless'
    ),
    'Circularity (whitened)': Metric(
        lambda ctx: ctx.compartment.whiten().isoperimetric_ratio(),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$\tilde{q}$',
        label_normal = r'$(\tilde{q}-\min(\tilde{q}))/(\overline{\tilde{q}}-\min(\tilde{q}))$',
        corr_transform = lambda x: x / PlanarPolygon.ir_ngon(6),
        corr_transform_label = r'$\tilde{q} / q_{\text{hex}}$',
        rvs = [],
        limits = (0, 1),
        units = 'unitless'
    ),
    'Offset': Metric(
        lambda ctx: la.norm(ctx.compartment.centroid() - ctx.cell.centroid()),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$r$',
        label_normal = r'$(r-\min(r))/(\overline{r}-\min(r))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(r)$',
        rvs = [stats.gamma],
        limits = (0, 10),
        units = r'$\mu m$'
    ),
    'Squared offset': Metric(
        lambda ctx: la.norm(ctx.compartment.centroid() - ctx.cell.centroid()) ** 2,
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$r^2$',
        label_normal = r'$(r^2-\min(r^2))/(\overline{r^2}-\min(r^2))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(r^2)$',
        rvs = [stats.gamma],
        limits = (0, 10),
        units = r'$\mu m^2$'
    ),
    'Offset angle': Metric(
        lambda ctx: angle_between_lines(ctx.cell.centroid() - ctx.compartment.centroid(), ctx.compartment.major_axis()) * 180 / np.pi,
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$\theta_{\rm cell}$',
        label_normal = r'$(\theta_{\rm cell}-\min(\theta_{\rm cell}))/(\overline{\theta_{\rm cell}}-\min(\theta_{\rm cell}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(\theta_{\rm cell})$',
        rvs = [stats.gamma],
        limits = (0, 90),
        units = r'degrees'
    ),
    'Offset (whitened)': Metric(
        lambda ctx: ctx.compartment.mahalanobis_distance(ctx.cell.centroid()),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$\tilde{r}$',
        label_normal = r'$(\tilde{r}-\min(\tilde{r}))/(\overline{\tilde{r}}-\min(\tilde{r}))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(\tilde{r})$',
        rvs = [],
        limits = (0, np.inf),
        units = 'unitless'
    ),
    'Circular radius': Metric(
        lambda ctx: ctx.compartment.circular_radius(),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$R$',
        label_normal = r'$(R-\min(R))/(\overline{R}-\min(R))$',
        corr_transform = lambda x: x,
        corr_transform_label = r'$R$',
        rvs = [],
        limits = (0, np.inf),
        units = r'$\mu m$'
    ),
    'Second moment': Metric(
        lambda ctx: ctx.compartment.trace_M2(ctx.compartment.centroid(), standardized=True),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$J$',
        label_normal = r'$(J-\min(J))/(\overline{J}-\min(J))$',
        corr_transform = lambda x: x,
        corr_transform_label = r'$J$'
    ),
    'Second moment about cell': Metric(
        lambda ctx: ctx.compartment.trace_M2(ctx.cell.centroid(), standardized=True),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min())
    ),
    'Voronoi error': Metric(
        lambda ctx: (1 - ctx.compartment.iou(ctx.voronoi)),
        normalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$e_V$',
        label_normal = r'$(e_V-\min(e_V))/(\overline{e_V}-\min(e_V))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(e_V)$',
        rvs = [],
        limits = (0, np.inf),
        units = 'unitless'
    ),
    'Voronoi neighbors': Metric(
        lambda ctx: ctx.voronoi.n,
        label = r'$n_V$',
        label_normal = r'$(n_V-\min(n_V))/(\overline{n_V}-\min(n_V))$',
        corr_transform = lambda x: np.log10(x),
        corr_transform_label = r'$\log_{10}(n_V)$',
        rvs = [],
        limits = (0, np.inf),
        units = 'count'
    ),
    'Distance to center': Metric(
        lambda ctx: la.norm(ctx.compartment.centroid() - ctx.boundary.v),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$d_{\rm center}$',
        label_normal = r'$(d_{\rm center}-\min(d_{\rm center}))/(\overline{d_{\rm center}}-\min(d_{\rm center}))$',
        corr_transform = lambda x: x,
        corr_transform_label = r'$d_{\rm center}$'
    ),
    'Mahalanobis distance to center': Metric(
        lambda ctx: ctx.boundary.mahalanobis_distance(ctx.compartment.centroid()),
        nondimensionalize = lambda xs: (xs - xs.min()) / (xs.mean() - xs.min()),
        label = r'$d_{\rm mah}$',
        label_normal = r'$(d_{\rm mah}-\min(d_{\rm mah}))/(\overline{d_{\rm mah}}-\min(d_{\rm mah}))$',
        corr_transform = lambda x: x,
        corr_transform_label = r'$d_{\rm mah}$'
    ),
}

def run():
   
    all_spheroid_metrics = pd.read_csv(f'{DATA_PATH}/topview/all_spheroid_metrics.csv')

    # for qualifier in ['', '.corrected']:
    for qualifier in ['']:
   
        all_raw_metrics = []
    
        def fun(stage: str, spheroid: int, path: str):
            boundary = pickle.load(open(f'{path}.boundary.rescaled{qualifier}', 'rb'))
            compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched{qualifier}', 'rb'))
            cells = pickle.load(open(f'{path}.cell_polygons.rescaled.matched{qualifier}', 'rb'))
            vors = pickle.load(open(f'{path}.voronoi.rescaled.matched{qualifier}', 'rb'))
            assert len(compartments) == len(cells) == len(vors)
            N = len(compartments)
            spheroid_metrics = all_spheroid_metrics[(all_spheroid_metrics['Stage'] == stage) & (all_spheroid_metrics['Spheroid'] == spheroid)]
            assert len(spheroid_metrics) == 1
            ap_vec = np.array([spheroid_metrics['AP X'].item(), spheroid_metrics['AP Y'].item()])

            columns = list(all_metrics.keys())
            column_vals = {
                'raw': dict(),
                'nondimensionalized': dict(),
                'normalized': dict(),
            }
            for col in columns:
                metric = all_metrics[col]
                column_vals['raw'][col] = np.array([metric(MetricContext(cell, compartment, voronoi, boundary, ap_vec)) for cell, compartment, voronoi in zip(cells, compartments, vors)])
                column_vals['nondimensionalized'][col] = metric.nondimensionalize(column_vals['raw'][col].copy())
                column_vals['normalized'][col] = metric.normalize(column_vals['nondimensionalized'][col].copy())

            for k, v in column_vals.items():
                df = pd.DataFrame(v, columns=columns)
                df.to_csv(f'{path}.polygon_metrics.{k}.csv', index=False)
                if k == 'raw':
                    v['Stage'] = [stage] * N
                    v['Spheroid'] = [spheroid] * N
                    df = pd.DataFrame(v)
                    all_raw_metrics.append(df)

        run_for_each_spheroid_topview(fun)

        all_raw_metrics = pd.concat(all_raw_metrics)
        all_raw_metrics.to_csv(f'{DATA_PATH}/topview/all_polygon_metrics.raw{qualifier}.csv', index=False)

def iterate_over_stage_metrics(fun: Callable[[str, pd.DataFrame], None], qualifier: str) -> pd.DataFrame:
    all_raw_metrics = pd.read_csv(f'{DATA_PATH}/topview/all_polygon_metrics.raw{qualifier}.csv')
    for stage in TOPVIEW_STAGES:
        df = all_raw_metrics[all_raw_metrics['Stage'] == stage]
        fun(stage, df)
    return all_raw_metrics

if __name__ == '__main__':
    run()
    # df = pd.read_csv(f'{DATA_PATH}/topview/all_polygon_metrics.raw.csv')
