# script to resize registered atlas and hemispheres
# processes aligned allen_10um atlas, output from brainreg_to_allen.sh

import pandas as pd
import numpy as np
from skimage.transform import resize
import tifffile as tf

home_dir = "/mnt/labNAS/"
# home_dir = "/Volumes/Data/"

meta_path=home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/aligned_file_meta_1b.csv"
metadata = pd.read_csv(meta_path, sep=" ")
# metadata = pd.read_csv("STP_metadata.csv")

# in files, out from brainreg_to_allen
straight_brain="allen_10um"
in_path=home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/"+straight_brain+"_aligned/"
out_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/input_tifs/resized_atlases/"

for i in range(metadata.shape[0]):
    print("working on", metadata.loc[i,"brain"])
    in_folder_path = in_path+metadata.loc[i,"brain"]+"_brainreg_"+straight_brain+"/"
    in_file_atlas = in_folder_path+"registered_atlas.tiff"
    in_file_hemis = in_folder_path+"registered_hemispheres.tiff"
    in_file_bounds = in_folder_path+"boundaries.tiff"

    # resize atlas/hemi file
    atlas = tf.imread(in_file_atlas)
    hemis = tf.imread(in_file_hemis)
    bounds = tf.imread(in_file_bounds)
    size = eval(metadata.loc[i,"rot_size"]) # read_csv reads in tuple as string, eval transforms it into tuple
    atlas_resize = resize(atlas, size,
                 mode='edge',
                 anti_aliasing=False,
                 anti_aliasing_sigma=None,
                 preserve_range=True,
                 order=0)
    hemis_resize = resize(hemis, size,
                 mode='edge',
                 anti_aliasing=False,
                 anti_aliasing_sigma=None,
                 preserve_range=True,
                 order=0)
    bounds_resize = resize(bounds, size,
                 mode='edge',
                 anti_aliasing=False,
                 anti_aliasing_sigma=None,
                 preserve_range=True,
                 order=0)
    
    # save resized images
    tf.imwrite(out_path+"allen_10um_to_"+metadata.loc[i, 'brain']+"_atlas_RESIZED.tiff", 
               atlas_resize)
    tf.imwrite(out_path+"allen_10um_to_"+metadata.loc[i, 'brain']+"_hemis_RESIZED.tiff", 
               hemis_resize)
    tf.imwrite(out_path+"allen_10um_to_"+metadata.loc[i, 'brain']+"_boundaires_RESIZED.tiff", 
               bounds_resize)