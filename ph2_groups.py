'''
Ph2 segmentations grouped by lifecycle stage
'''

from typing import Dict, List, Generator, Any
import pickle
import os
from tqdm import tqdm
import sys

from ph2_segmentation import *
from asrvsn_mpl import *
import microseg
import microseg.data
import microseg.data.seg_2d
import matgeo
import matgeo.plane
import matgeo.ellipsoid

class GroupedPh2Segmentations:
    '''
    PhII segmentations grouped by lifecycle stage
    '''
    def __init__(
            self, 
            groups: Dict[int, List[Ph2Segmentation]],
            names: Dict[str, int], # Names must be unique, indexing groups
        ):
        self.groups = groups
        self.names = names
        assert set(names.values()) == set(groups.keys()), 'Names must index groups'
        self.names_reverse = {v: k for k, v in names.items()}

        ## Derived stuff
        self.ordering = {group: GroupedPh2Segmentations.ordering_key(name, group, groups[group]) for name, group in names.items()}
        self.legend_names = {group: f'{name} (n={len(groups[group])})' for name, group in names.items()}
        self.colors = { 
            'I': 'blue', # map_color_01(0, cc.bgy), # Corresponding to wall-clock time, "green" for algae :)
            'II': 'red', # map_color_01(15/48, cc.bgy),
            'III': 'magenta', # map_color_01(21/48, cc.bgy),
            'IV': 'darkgreen', # map_color_01(37/48, cc.bgy),
            'S': 'darkorange',
        }
        self.colors = {v: self.colors[k] for k, v in self.names.items()}

        ## Pick a segmentation to display from the last group
        self.display_group = [k for k, _, __, ___ in self.iterate()][-1]
        self.display_seg = self.groups[self.display_group][0]

    def __len__(self) -> int:
        return len([x for x in self.all_segmentations])

    def iterate(self) -> Generator:
        '''
        Iterate over segmentations in group-sorted order
        '''
        for k in sorted(list(self.groups.keys()), key=lambda x: self.ordering[x]):
            yield k, self.groups[k], self.names_reverse[k], self.legend_names[k]

    def __getitem__(self, key: int) -> List[Ph2Segmentation]:
        '''
        Return group by index
        '''
        return self.groups[key]

    def apply_group_legend(self, ax: plt.Axes, **kwargs) -> plt.Axes:
        '''
        Apply legend to axis
        '''
        pt.ax_color_legend(ax, self.ordered_names, self.ordered_colors, **kwargs)

    def get_by_name(self, name: str) -> List[Ph2Segmentation]:
        '''
        Get segmentations by name
        '''
        return self.groups[self.names[name]]
    
    def get_by_pathname(self, group_name: str, pathname: str) -> Ph2Segmentation:
        grp = self.get_by_name(group_name)
        segs = [seg for seg in grp if seg.pathname == pathname]
        assert len(segs) == 1, f'Zero or multiple segmentations with pathname {pathname}'
        return segs[0]

    def recompute(self):
        '''
        Recompute all segmentations
        '''
        for seg in tqdm(list(self.all_segmentations)):
            seg.recompute()
        ## Normalized metrics
        self.normalized_metrics = {
            name: np.concatenate([
                # Normalize across all segmentations
                metric.normalize(seg.metrics[name]['nondim']) for seg in self.all_segmentations 
            ])
            for name, metric in Ph2Segmentation.computed_metrics.items()
        }

    @property
    def ordered_names(self) -> List[str]:
        '''
        Return ordered names
        '''
        return [name for _, __, name, ___ in self.iterate()]

    @property
    def ordered_colors(self) -> List[str]:
        '''
        Return ordered colors
        '''
        return [self.colors[k] for k, _, __, ___ in self.iterate()]

    @property
    def n_groups(self) -> int:
        '''
        Number of groups
        '''
        return len(self.groups)

    @property
    def all_segmentations(self) -> Generator:
        '''
        All segmentations
        '''
        for _, segs, __, ___ in self.iterate():
            for seg in segs:
                yield seg

    @property
    def n_segmentations(self) -> int:
        '''
        Number of total segmentations
        '''
        return len(list(self.all_segmentations))

    @staticmethod
    def ordering_key(name: str, group: int, segs: List[Ph2Segmentation]) -> Any:
        '''
        Ordered value for sorting groups (e.g. by size)
        '''
        # return np.mean([seg.seg_metrics['Circular radius'] for seg in segs]) # Average circular radius
        return name

    @staticmethod
    def from_folder(folder: str, recompute: bool=False) -> 'GroupedPh2Segmentations':
        '''
        Load segmentations from folder
        '''
        folder = os.path.abspath(folder)
        print(f'Segmenting data in folder {folder}')
        assert os.path.isdir(folder), f'Folder {folder} does not exist'
        state = {
            'segs': [],
            'groups': [],
            'names': [],
            'c_n': 0, 
        }
        def get_segmentations_in_folder(folder: str):
            print(f'Processing folder {folder}')
            folder_group = state['c_n']
            state['c_n'] += 1
            names_in_folder = set([f.split('.')[0] for f in os.listdir(folder) if not f.startswith('.')])
            folder_name = os.path.basename(folder)
            if folder_name != 'excluded':
                # Monkey-patch modules for Segmentation2D unpickling
                sys.modules['seg_2d'] = microseg.data.seg_2d
                sys.modules['plane'] = matgeo.plane
                sys.modules['ellipsoid'] = matgeo.ellipsoid

                for name in names_in_folder:
                    f = os.path.join(folder, name)
                    f_seg = os.path.join(folder, f'{name}.seg')
                    f_seg_ph2 = os.path.join(folder, f'{name}.seg_ph2')
                    f_seg_gon = os.path.join(folder, f'{name}.ant_gonidia')
                    f_seg_off = os.path.join(folder, f'{name}.offspring')

                    if os.path.isfile(f_seg):
                        print(f'Found .seg file: {f_seg}, processing...')
                        assert os.path.isfile(f_seg_gon), f'No .ant_gonidia file at {f_seg_gon}'
                        anterior_offspring = pickle.load(open(f_seg_gon, 'rb'))
                        assert os.path.isfile(f_seg_off), f'No .offspring file at {f_seg_off}'
                        all_offspring = [[roi.asPoly() for roi in sublist] for sublist in pickle.load(open(f_seg_off, 'rb'))]
                        if os.path.isfile(f_seg_ph2) and (not recompute):
                            print(f'Found cached .seg_ph2 file: {f_seg_ph2}, not recomputing...')
                            seg = pickle.load(open(f_seg_ph2, 'rb'))
                            print(type(seg)) # Needed but not sure why.
                            # assert type(seg) == Ph2Segmentation
                            state['segs'].append(seg)
                            state['groups'].append(folder_group)
                            state['names'].append(folder_name)
                        elif os.path.isfile(f_seg):
                            print(f'Computing .seg_ph2 file: {f_seg_ph2}...')
                            seg = pickle.load(open(f_seg, 'rb'))
                            img_path = os.path.join(folder, f'{name}.czi')
                            seg = Ph2Segmentation(seg, anterior_offspring, all_offspring, name, img_path)
                            seg.recompute()
                            print('Computed polygons & metrics')
                            state['segs'].append(seg)
                            state['groups'].append(folder_group)
                            state['names'].append(folder_name)
                            pickle.dump(seg, open(f_seg_ph2, 'wb'))
                            print(f'Cached: {f_seg_ph2}')
                    elif os.path.isdir(f):
                        get_segmentations_in_folder(f)
                    else:
                        print(f'Unrecognized file: {f}, skipping...')

        get_segmentations_in_folder(folder)
        groups = {k: [seg for seg, group in zip(state['segs'], state['groups']) if group == k] for k in set(state['groups'])}
        names = dict(zip(state['names'], state['groups']))
        return GroupedPh2Segmentations(groups, names)

