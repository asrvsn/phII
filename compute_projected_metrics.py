'''
Comparison of metrics computed on projected and unprojected polygons
'''

from dataclasses import dataclass
from lib import *
from compute_polygon_metrics import all_metrics

@dataclass
class ProjectedMetricContext:
    '''
    A subset of the MetricContext which can be computed for the projected features
    '''
    compartment: PlanarPolygon

    def get_ap_value(self, pos: np.ndarray) -> float:
        raise NotImplementedError

def run():
    all_proj_metrics = []

    def fun(stage: str, spheroid: int, path: str):
        proj_polys_path = f'{path}.compartment_polygons.rescaled.matched.projected'
        if os.path.exists(proj_polys_path):
            print(f'Computing projected metrics for {path}')
            polys = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
            proj_polys = pickle.load(open(proj_polys_path, 'rb'))
            assert len(proj_polys) > 0, f'No projected polygons found for {path}'
            assert len(proj_polys) == len(polys), f'Number of projected and unprojected polygons do not match for {path}'
            N = len(proj_polys)
            # Get the metrics which can be computed for the projected features
            columns = []
            ctx = ProjectedMetricContext(proj_polys[0])
            for col in all_metrics.keys():
                try:
                    value = all_metrics[col](ctx)
                    columns.append(col)
                except NotImplementedError:
                    # Skip metrics that rely on unimplemented methods
                    pass
                except AttributeError as e:
                    # Skip metrics that require attributes not available in ProjectedMetricContext
                    pass
            column_vals = dict()
            column_vals['Stage'] = [stage] * N
            column_vals['Spheroid'] = [spheroid] * N
            for col in columns:
                column_vals[col] = np.array([
                    all_metrics[col](ProjectedMetricContext(poly)) for poly in proj_polys
                ])
            df = pd.DataFrame(column_vals)
            df.to_csv(f'{path}.projected_metrics.csv', index=False)
            all_proj_metrics.append(df)

    run_for_each_spheroid_topview(fun)
    
    all_proj_metrics = pd.concat(all_proj_metrics)
    all_proj_metrics.to_csv(f'{DATA_PATH}/topview/all_projected_metrics.csv', index=False)

def iterate_over_stage_metrics_projected(fun: Callable[[str, pd.DataFrame, pd.DataFrame], None], *args) -> pd.DataFrame:
    all_raw_metrics = pd.read_csv(f'{DATA_PATH}/topview/all_polygon_metrics.raw.csv')
    proj_metrics = pd.read_csv(f'{DATA_PATH}/topview/all_projected_metrics.csv')
    datasource = []
    for stage in TOPVIEW_STAGES:
        df_raw = all_raw_metrics[all_raw_metrics['Stage'] == stage]
        df_proj = proj_metrics[proj_metrics['Stage'] == stage]
        if len(df_proj) > 0: # Only those stages for which we have data
            spheroids_in_proj = df_proj['Spheroid'].unique()
            df_raw = df_raw[df_raw['Spheroid'].isin(spheroids_in_proj)]
            assert np.array_equal(df_raw['Spheroid'].values, df_proj['Spheroid'].values)
            assert len(df_raw) == len(df_proj) 
            assert len(df_raw) > 0
            fun(stage, df_raw, df_proj, *args)
            df_proj_renamed = df_proj.rename(columns={col: f"{col} (projected)" for col in df_proj.columns})
            df = pd.concat([df_raw.reset_index(drop=True), df_proj_renamed.reset_index(drop=True)], axis=1)
            datasource.append(df)

    datasource = pd.concat(datasource)
    return datasource

if __name__ == '__main__':
    run()
