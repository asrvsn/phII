from lib import *

def fun(stage, path):
    # Renaming some files
    # tif_path = f'{path}.tif'
    # tif_name = os.path.basename(tif_path)
    # new_tif_name = tif_name.split(' ')[-1]
    # new_tif_path = os.path.dirname(tif_path) + '/' + new_tif_name
    # os.rename(tif_path, new_tif_path)
    pass

if __name__ == '__main__':
    run_for_each_spheroid_crosssection(fun)
