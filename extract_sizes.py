from lib import *

def run():
    sizes = dict()

    def fun(stage, spheroid, path):
        boundary = pickle.load(open(f'{path}.boundary.rescaled', 'rb'))
        offspring = pickle.load(open(f'{path}.offspring.all.rescaled', 'rb'))
        cells = pickle.load(open(f'{path}.cell_polygons.rescaled', 'rb'))
        if not stage in sizes:
            sizes[stage] = {
                'Spheroid radii': [],
                'Offspring radii': [],
                'Somatic cell radii': [],
            }
        sizes[stage]['Spheroid radii'].append(boundary.circular_radius())
        sizes[stage]['Offspring radii'].extend([p.circular_radius() for p in offspring])
        sizes[stage]['Somatic cell radii'].extend([p.circular_radius() for p in cells])

    run_for_each_spheroid_topview(fun)
    pickle.dump(sizes, open(f'{DATA_PATH}/topview/sizes.pkl', 'wb'))

if __name__ == '__main__':
    run()