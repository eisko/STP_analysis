# STP_analysis_process.py
# functions for processing aligned STP data

import pandas as pd
import numpy as np
# using skimage
from skimage import io # import tiff file as ndarray
from skimage.segmentation import find_boundaries # for generating boundaries
import os
# from make_masks import areas

# areas to generate masks for
areas = ["grey", "CTX", "TH", "STR", "CP", "P", "MB", "PAG", "SCm", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU"]



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
    
def make_boundaries(plot_areas, mask_list, mask_list_order=areas, roi=None, slice=None, scaling_factor=1000, boundary_mode="thick"):
    """_summary_

    Args:
        plot_areas (list): List of strings of areas in order to include to draw boundaries
        mask_list (list): List of all masks to draw from to make boundaries.
        mask_list_order (list, optional): List of strings of corresponding area labels for mask in mask_list
                                        Defaults to areas
        roi (int, optional): used to determine bounds for max slices. Defaults to None.
        slice (int, optional): Set to slice index if want to return 1 slice, or slice of slices. Defaults to None.
        scaling_factor (int, optional): how much to scale the final boundary value (i.e. intensity) to make visible when plotting.
                                Defaults to 1000
        boudnary_mode (str, optional): mode used to make boundaries, inherited from skimage.segmentation.find_boudnaries
                            Defaults to "thick"

    Returns:
        _type_: _description_
    """
    # 0.
    # create masks, which is list containing on masks to be included in output
    masks = [mask_list[mask_list_order.index(plot_areas[i])] for i in range(len(plot_areas))]

    # 1. determine slices to return
    # if roi == int, roi==slice, slice==true:
    # output idx = slice bounds?
    if slice:
        if len(slice)==1:
            start = slice
            end = slice + 1
        elif len(slice)==2:
             start = slice[0]
             end = slice[1]
    elif roi:
            # get start/end of roi
            roi_mask = mask_list[mask_list_order.index(roi)]
            roi_idx = roi_mask.sum(axis=1).sum(axis=1) > 0
            slices_n = np.array(range(roi_idx.shape[0]))
            start = slices_n[roi_idx].min()
            end = slices_n[roi_idx].max()
    else:
        start = 0
        end = 201

    # set bounds for every mask
    masks = [m[start:end] for m in masks]

    # 2. max project all masks after setting bounds
    masks_max = [m.max(axis=0) for m in masks]

    # 3. add masks on top of each other -> can't do this or will also get boundaries of intersections of masks
    # 3. instead, 
    #   3a) set each mask as diff number in order specified in areas_list
    masks_id = [masks_max[i]*(i+1) for i in range(len(masks_max))]
    #   3b) take maximum of all masks
    mask_final = masks_id[0] # initailize
    for i in range(1,len(masks_id)):
        mask_final = np.maximum(mask_final, masks_id[i]) # get maximum based on next mask

    # could return(mask_final) to visualize w/ matplotlib, where area labelled by color
    # return(mask_final)

    # 4. find boundaries w/ skimage?
    boundaries = find_boundaries(mask_final.astype(int), mode=boundary_mode).astype(
        np.int8, copy=False)

    # 5. convert image to 0,1*scaling_factor???
    boundaries_scaled = boundaries * scaling_factor


    return(boundaries_scaled)

def calc_fluor(images, metadata, mm_masks, st_masks, mask_areas, areas_to_plot):
    """Calculates integrated fluorescence on a per area basis.
    Returns a pandas dataframe

    Args:
        images (list): list of images to calculate
        metadata (dataframe): Pandas dataframe where each row corresponds to image in iamges
        mm_masks (list): mmus masks from mmus aligned atlas
        st_masks (list): steg masks from steg aligned atlas
        mask_areas (list): List of strings that label each mask in mm/st_masks
        areas_to_plot (list): List of strings for each area to plot
    """

    # initialize df
    fluor_df = pd.DataFrame(columns=["area", "Integrated Fluorescence", "Volume_mm3", "brain", "species", "inj_site"])

    for i in range(len(images)):
        row_met = metadata.iloc[i,:]

        if row_met["species"]=="STeg":
            mask_list = st_masks
        elif row_met["species"]=="MMus":
            mask_list = mm_masks
        print(row_met)

        for j in range(len(areas_to_plot)):
            area_idx = mask_areas.index(areas_to_plot[j])
            print(areas_to_plot[j], area_idx)
            mask = mask_list[area_idx]
            roi = np.multiply(images[i], mask)
            fluor = roi.sum() * 0.00001 # 10^-6 scaling factor
            vol = mask.sum() * 0.02 # mm^3 per voxel
            row = {"area":mask_areas[j],
                    "Fluorescence": fluor,
                    "Volume_mm3": vol,
                    "brain":row_met["brain"],
                    "species":row_met["species"],
                    "inj_site":row_met["inj_site"]}
            fluor_df.loc[len(fluor_df.index)] = row
    
    return(fluor_df)
