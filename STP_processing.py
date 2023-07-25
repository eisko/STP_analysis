# STP_analysis_process.py
# functions for processing aligned STP data

import pandas as pd
import numpy as np
# using skimage
from skimage import io # import tiff file as ndarray
import os

def make_mask(roi, atlas_in):
    '''
    Funtion to make mask
    Check if input roi in list of structure acronyms
    returns ndarray of 0/1 of whether pixel origin in roi

    Takes roi and registered atlas as input:
        roi = area string, e.g. 'PAG'
        atlas =  path string e.g. "/path/to/registered_atlas.tif"
    '''
    
    # check which computer being used to call
    home = os.path.expanduser('~')
    
    if home == '/Users/eisko':
        # for home mac
        structures = pd.read_csv('/Users/eisko/.brainglobe/allen_mouse_25um_v1.2/structures.csv')
    elif home == '/home/emily':
        # for acadia
        structures = pd.read_csv('/home/emily/.brainglobe/allen_mouse_25um_v1.2/structures.csv')

    # check if atlas is path (aka string)
    if type(atlas_in) == str:
        # import atlas from given path
        atlas = io.imread(atlas_in, plugin="tifffile")
    else:
        atlas = atlas_in

    dims = atlas.shape

    # check if roi in structure list
    if roi in structures.loc[:, 'acronym'].tolist():
        # extract id of region of interest (roi)
        roi_bool = structures.loc[:, 'acronym']== roi
        roi_row = structures.loc[roi_bool,:]
        roi_id = roi_row.iloc[0]['id']

        # get all sub-regions relevant to this region
        childs_bool = structures.structure_id_path.str.contains('/'+str(roi_id)+'/')
        childs = structures.loc[childs_bool, :]
        child_ids = childs.id.tolist()

        # 3. create mask for roi by summing mask from all subregions
        # mask = atlas==roi_id
        mask_out = np.zeros(dims)
        for child in child_ids:
            mask = atlas==child
            mask_out = np.add(mask_out, mask)

        return(mask_out)