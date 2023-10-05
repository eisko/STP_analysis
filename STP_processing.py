# STP_analysis_process.py
# functions for processing aligned STP data

import pandas as pd
import numpy as np
# using skimage
from skimage import io # import tiff file as ndarray
from skimage.segmentation import find_boundaries # for generating boundaries
import os
from scipy import stats
import math

# from make_masks import areas

# areas to generate masks for
areas = ["grey", "CTX", "OMCc", "ACAc", "aud","TH", "STR", "CP", "AMY", "P", "PG", "MB", "PAG", "SCm", 
         "SNr", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU", "BS"]


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
    
def make_boundaries(plot_areas, mask_list, mask_list_order=areas, roi=None, slice=None, 
                    scaling_factor=1000, boundary_mode="thick"):
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
        end = mask_list[0].shape[0]

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

def make_boundaries_dict(plot_areas, mask_dict, roi=None, slice=None, 
                         scaling_factor=1000, boundary_mode="thick", view="front"):
    """_summary_

    Args:
        plot_areas (list): List of strings of areas in order to include to draw boundaries
        mask_dict (dict): dictionary where keys are areas and values are the corresponding mask
        roi (int, optional): used to determine bounds for max slices. Defaults to None.
        slice (int, optional): Set to slice index if want to return 1 slice, or slice of slices. Defaults to None.
        scaling_factor (int, optional): how much to scale the final boundary value (i.e. intensity) to make visible when plotting.
                                Defaults to 1000
        boudnary_mode (str, optional): mode used to make boundaries, inherited from skimage.segmentation.find_boudnaries
                            Defaults to "thick"
        view (str, optional): If given, transpose masks to match view, can be ["top", "side", "front"]. Defaults to front.

    Returns:
        _type_: _description_
    """
    # # 0.
    # # create masks, which is list containing on masks to be included in output
    # masks = [mask_list[mask_list_order.index(plot_areas[i])] for i in range(len(plot_areas))]

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
            roi_mask = mask_dict[roi]
            roi_idx = roi_mask.sum(axis=1).sum(axis=1) > 0
            slices_n = np.array(range(roi_idx.shape[0]))
            start = slices_n[roi_idx].min()
            end = slices_n[roi_idx].max()
    else:
        start = 0
        end = mask_dict[plot_areas[0]].shape[0]

    # set order for masks
    masks = [mask_dict[area] for area in plot_areas]

    # set bounds for every mask
    masks = [m[start:end] for m in masks]


    # transpose masks 
    if view=="front":
        transform = (0,1,2)
    elif view=="side":
        transform = (2,1,0)
    elif view=="top":
        transform = (1,0,2)

    masks = [np.transpose(m, transform) for m in masks]




    # 2. max project all masks after setting bounds
    masks_max = [m.max(axis=0) for m in masks]

    # 3. add masks on top of each other -> can't do this or will also get boundaries of intersections of masks
    # 3. instead, 
    #   3a) set each mask as diff number in order specified in areas_list
    masks_id = [masks_max[i]*(i+1) for i in range(len(masks_max))]
    #   3b) take maximum of all masks
    mask_final = masks_id[0] # initailize
    for i in range(len(masks_id)):
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
    fluor_df = pd.DataFrame(columns=["area", "Fluorescence", "Volume_mm3", "brain", "species", "inj_site"])

    for i in range(len(images)):
        row_met = metadata.iloc[i,:]

        if row_met["species"]=="STeg":
            mask_list = st_masks
        elif row_met["species"]=="MMus":
            mask_list = mm_masks

        for j in range(len(areas_to_plot)):
            area_idx = mask_areas.index(areas_to_plot[j])
            mask = mask_list[area_idx]
            roi = np.multiply(images[i], mask)
            fluor = roi.sum() * 0.00001 # 10^-6 scaling factor
            vol = mask.sum() * 0.02 # mm^3 per voxel
            row = {"area":areas_to_plot[j],
                    "Fluorescence": fluor,
                    "Volume_mm3": vol,
                    "brain":row_met["brain"],
                    "species":row_met["species"],
                    "inj_site":row_met["inj_site"]}
            fluor_df.loc[len(fluor_df.index)] = row
    
    return(fluor_df)

def calc_fluor_individ(images, metadata, mask_dict, areas_to_plot, 
                       species=None, inj_site=None, int_fluor_scale=0.00001,
                       ):
    """Calculates integrated fluorescence on a per area basis.
    Returns a pandas dataframe

    Args:
        images (list): list of images to calculate
        metadata (dataframe): Pandas dataframe where each row corresponds to image in iamges
        mask_dict (dictionary): dictionary where keys are areas and values are masks in same order as metadata
        areas_to_plot (list): List of strings for each area to plot
        species (string, optional): What species to return. Defaults to None.
        inj_site (string, optional): What injection site to return. Defaults to None.
    """

    # initialize df
    fluor_df = pd.DataFrame(columns=["area", "Fluorescence", "Volume_mm3", "brain", "species", "inj_site"])

    if species:
        metadata = metadata[metadata["species"]==species]
    if inj_site:
        metadata = metadata[metadata["inj_site"]==inj_site]
    
    meta_idx = list(metadata.index)

    for i in meta_idx:
        row_met = metadata.loc[i,:]

        for j in range(len(areas_to_plot)):
            mask = mask_dict[areas_to_plot[j]][i]
            roi = np.multiply(images[i], mask)
            fluor = roi.sum() * int_fluor_scale # 10^-6 scaling factor -> to make reasonable?
            vol = mask.sum() * 0.02 # mm^3 per voxel
            row = {"area":areas_to_plot[j],
                    "Fluorescence": fluor,
                    "Volume_mm3": vol,
                    "brain":row_met["brain"],
                    "species":row_met["species"],
                    "inj_site":row_met["inj_site"]}
            fluor_df.loc[len(fluor_df.index)] = row
    
    return(fluor_df)

def normalize_by_area(df_fluor, norm_area):
    """
    Takes in output from calc_fluor_individ and returns normalized flourescence column

    Args:
        df_fluor (DataFrame): output from calc_fluor_individ()
        norm_area (string): are to use as normalization factor
    """

    output = pd.DataFrame()

    for b in df_fluor['brain'].unique():
        b_fluor = df_fluor[df_fluor['brain']==b]
        b_area = b_fluor[b_fluor['area']==norm_area]
        b_area_value = b_area['Fluorescence'].values[0]
        b_normed = b_fluor['Fluorescence']/b_area_value
        b_fluor['normalized_fluorescence'] = b_normed
        output = pd.concat([output, b_fluor])

    return output

def df_ttest(df, test_vals="Fluorescence", compare_group="species",
             group1="STeg", group2="MMus"):
    """output dataframe based on comparison of species proportional means
        output dataframe can be used for making volcano plot

    Args:
        df (pd.DataFrame): output of dfs_to_proportions
        test_vals (str, optional): name of column to do ttest comparison.
                                    Defaults to "Fluorescence".
    """

    areas = sorted(df['area'].unique())

    # for area in areas:
    #     area_df = df[df['area']==area]
    #     mean = df.groupby('area', sort = False, as_index=False)['proportion'].mean()

    g1_df = df[df[compare_group]==group1]
    g1_array = g1_df.pivot(columns='brain', values=test_vals, index='area').values

    g2_df = df[df[compare_group]==group2]
    g2_array = g2_df.pivot(columns='brain', values=test_vals, index='area').values

    results = stats.ttest_ind(g1_array, g2_array, axis=1)
    p_vals = results[1]
    plot = pd.DataFrame({"area":areas, "p-value":p_vals})
    plot[group1+"_mean"] = g1_array.mean(axis=1)
    plot[group2+"_mean"] = g2_array.mean(axis=1)
    # plot["effect_size"] = (plot["st_mean"]-plot["mm_mean"]) / (plot["st_mean"] + plot["mm_mean"]) # modulation index
    plot["fold_change"] = plot[group1+"_mean"]/(plot[group2+"_mean"])
    plot["log2_fc"] = np.log2(plot["fold_change"])
    plot["nlog10_p"] = -np.log10(plot["p-value"])

    return(plot)


def compare_groups(data, group1="MMus", group2="STeg", compare_group="species",
                   to_compare="Fluorescence", label="inter"):
    """Given data calculate differences between individual points among/within species,
    on a per area basis

    Args:
        data (DataFrame): pd.DataFrame, output of dfs_to_proportions()
        group1 (str, optional): Species to compare to. Defaults to "MMus".
        group2 (str, optional): Other species to compare. Defaults to "STeg".
        compare_group (str, optional): column that contains groups to compare. Defaults to "species".
        to_compare (str, optional): value to compare. Defaults to "Fluorescence".
        label (str, optional): Label of intra/inter comparison. Defautls to "inter".
    """

    df1 = data[data[compare_group]==group1]
    df2 = data[data[compare_group]==group2]

    areas = data['area'].unique()

    out_df = pd.DataFrame(columns=['area', to_compare+'_diff', compare_group, 'brain', 'label'])

    for area in areas:
        
        df1a = df1[df1['area']==area].reset_index(drop=True)
        df2a = df2[df2['area']==area].reset_index(drop=True)
        
        for a in range(df1a.shape[0]):
            for b in range(df2a.shape[0]):
                diff = df1a.loc[a, to_compare] - df2a.loc[b, to_compare]
                abs_diff = math.sqrt(diff**2)
                row = [area, abs_diff, group1+"_"+group2, df1a.loc[a, 'brain']+"_"+df2a.loc[b,'brain'], label]
                i = out_df.shape[0]
                out_df.loc[i, :] = row
                
    return(out_df)