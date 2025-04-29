'''
Export raw images to Zenodo
'''

import os
import shutil
from lib import *

def export_topview(stage: str, spheroid: int, path: str):
    '''
    Export a CZI file to Zenodo
    '''
    in_path = f'{path}.czi'
    out_path = f'{ZENODO_PATH}/Raw_images/topview/{stage}/spheroid_{spheroid}.czi'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy(in_path, out_path)

def export_crosssection(stage, path):
    '''
    Export tif cross section to Zenodo
    '''
    in_path = f'{path}.tif'
    out_path = f'{ZENODO_PATH}/Raw_images/crosssection/Stage_{stage}.tif'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy(in_path, out_path)

if __name__ == '__main__':
    run_for_each_spheroid_topview(export_topview)
    run_for_each_spheroid_crosssection(export_crosssection)