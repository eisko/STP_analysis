import numpy as np
import pandas as pd
import tifffile as tf

# take aligned files, subtract mean background intensity value
home_dir = "/mnt/labNAS/"
# home_dir = "/Volumes/Data/"

# import file with file names/paths
p05_paths = pd.read_csv(home_dir+"Emily/STP_for_MAPseq/processed_data/file_path.csv", names=["brain", "path"])
means = pd.read_csv(home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/aligned_file_meta.csv", sep=" ")
meta = means.merge(p05_paths,on="brain")


out_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/input_tifs/norm_subtracted/p05_mean_subtracted/"


n_files = meta.shape[0]

for i in range(n_files):
    print('working on', meta.loc[i, "brain"])
    # read in file
    # print(meta.loc[i, "path"])
    im = tf.imread(home_dir+meta.loc[i, "path"])
    # subtract mean value
    im_sub = im - meta.loc[i, "mean_subtract"]

    # sets negative numbers as +2^16, so set these as 0
    im_sub[im_sub > 60000] = 0

    # rotate 90 degrees so same orientation as atlas
    im_rot = np.rot90(im_sub, k=-1, axes=(1,2))

    # also flip left/right to get it in proper orientation
    im_flip = np.flip(im_rot, axis=2)


    # save as norm_subtracted files
    # print(out_path+meta.loc[i, 'brain']+"p05_norm_subtracted.tif")
    # for some reason doesn't work if imagej=True
    tf.imwrite(out_path+meta.loc[i, 'brain']+"_p05_norm_subtracted.tif", im_flip) 
    print("\n")

