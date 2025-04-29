'''
Summarize segmentation
'''

from lib import *

def run():
    rows = []
    columns = ['Stage', 'Spheroid', 'Circular radius', 'Segmented cell/compartment pairs']
    def fun(stage, spheroid, path):
        boundary = pickle.load(open(f'{path}.boundary.rescaled', 'rb'))
        cells = pickle.load(open(f'{path}.cell_polygons.rescaled.matched', 'rb'))
        compartments = pickle.load(open(f'{path}.compartment_polygons.rescaled.matched', 'rb'))
        assert len(cells) == len(compartments)
        row = [stage, spheroid, int(round(boundary.circular_radius(), 0)), len(cells)]
        rows.append(row)
    run_for_each_spheroid_topview(fun)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f'{DATA_PATH}/topview/segmentation_summary.csv')

if __name__ == '__main__':
    run()