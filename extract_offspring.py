'''
Extract offspring 
'''

from lib import *
from matgeo.plane import PlanarPolygonPacking
from microseg.utils.data import get_voxel_size
from asrvsn_math.array import flatten_list

# Parameters

def extract_offspring(stage: str, spheroid: int, path: str):
    # All offspring
    # # Migration
    # offspring = pickle.load(open(f'{path}.offspring', 'rb'))
    # offspring = [
    #     [roi.asPoly().migrate_OLD() for roi in sublist] for sublist in offspring
    # ]
    # pickle.dump(offspring, open(f'{path}.offspring', 'wb'))
    ## Flatten
    offspring = pickle.load(open(f'{path}.offspring', 'rb'))
    offspring = flatten_list(offspring)
    pickle.dump(offspring, open(f'{path}.offspring.all', 'wb'))
    ## Rescale
    voxsize = get_voxel_size(f'{path}.czi', fmt='XY')
    offspring = [p.rescale(voxsize) for p in offspring]
    pickle.dump(offspring, open(f'{path}.offspring.all.rescaled', 'wb'))
    # Anterior offspring
    ## Migration
    # ant_offspring = [p.migrate_OLD() for p in pickle.load(open(f'{path}.ant_gonidia', 'rb'))]
    # pickle.dump(ant_offspring, open(f'{path}.offspring.anterior', 'wb'))
    # os.remove(f'{path}.ant_gonidia')
    ## Rescale
    ant_offspring = pickle.load(open(f'{path}.offspring.anterior', 'rb'))
    ant_offspring = [p.rescale(voxsize) for p in ant_offspring]
    pickle.dump(ant_offspring, open(f'{path}.offspring.anterior.rescaled', 'wb'))

if __name__ == '__main__':
    run_for_each_spheroid_topview(extract_offspring)