# script to resize registered atlas and hemispheres
# processes aligned allen_10um atlas, output from brainreg_to_allen.sh

import pandas as pd
import numpy as np
from skimage.transform import resize
import tifffile as tf


csv_file="stp_metadata.csv"
metadata = pd.read_csv(csv_file)
# in files, out from brainreg_to_allen
straight_brain="allen_10um"
in_path="/mnt/labNAS/Emily/STP_for_MAPseq/3_brainreg_output/"+straight_brain+"_aligned/"
out_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/resized_atlases/"

for i in range(metadata.shape[0]):
    print("working on", metadata.loc[i,"brain"])
    in_folder_path = in_path+metadata.loc[i,"brain"]+"_brainreg_"+straight_brain+"/"
    in_file_atlas = in_folder_path+"registered_atlas.tiff"
    in_file_hemis = in_folder_path+"registered_hemispheres.tiff"

    # resize atlas/hemi file
    atlas = tf.imread(in_file_atlas)
    hemis = tf.imread(in_file_hemis)
    atlas_resize = resize(atlas, (265, 499, 640),
                 mode='edge',
                 anti_aliasing=False,
                 anti_aliasing_sigma=None,
                 preserve_range=True,
                 order=0)
    hemis_resize = resize(hemis, (265, 499, 640),
                 mode='edge',
                 anti_aliasing=False,
                 anti_aliasing_sigma=None,
                 preserve_range=True,
                 order=0)
    
    # save resized images
    tf.imwrite(out_path+"allen_10um_to_"+metadata.loc[i, 'brain']+"_atlas_RESIZED.tif", 
               atlas_resize, 
               imagej=True)
    tf.imwrite(out_path+"allen_10um_to_"+metadata.loc[i, 'brain']+"_hemis_RESIZED.tif", 
               hemis_resize, 
               imagej=True)