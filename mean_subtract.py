import numpy as np
import pandas as pd
import tifffile as tf

# take aligned files, subtract mean background intensity value

# import file with file names/paths
info = pd.read_csv("/Volumes/Data/Emily/STP_for_MAPseq/3_brainreg_output/aligned_file_paths.csv")

align_brain = info.loc[0,'align_to']
out_path = "/Volumes/Data/Emily/STP_for_MAPseq/4_python_output/input_tifs/norm_subtracted/"+align_brain+"_aligned/"


n_files = info.shape[0]

for i in range(n_files):
    print('working on', info.loc[i, "brain"])
    # read in file
    im = tf.imread(info.loc[i, "path"])
    # subtract mean value
    im_sub = im - info.loc[i, "mean_subtract"]

    # sets negative numbers as +2^16, so set these as 0
    im_sub[im_sub > 60000] = 0

    # save as norm_subtracted files
    tf.imwrite(out_path+info.loc[0, 'brain']+'_aligned_to_'+info.loc[0,'align_to']+"_norm_subtracted.tif", im, 
               imagej=True)

