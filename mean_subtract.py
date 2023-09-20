import numpy as np
import pandas as pd
import tifffile as tf

# take aligned files, subtract mean background intensity value
align_brain = "MMus_220303"

# define home dir
home_dir = "/mnt/labNAS/"

# import file with file names/paths
# meta = pd.read_csv(home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/aligned_file_paths.csv")
metadata = pd.read_csv("stp_metadata.csv")

# align_brain = meta.loc[0,'align_to']
in_path = home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/"+align_brain+"_aligned/"
out_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/input_tifs/norm_subtracted/"+align_brain+"_aligned/"


# import file with file names/paths
means = pd.read_csv(home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/aligned_file_meta.csv", sep=" ")
meta = means.merge(metadata,on="brain")

n_files = meta.shape[0]

for i in range(n_files):
    print('working on', meta.loc[i, "brain"])
    
    ########### MUST CHANGE BASED ON ALIGNED BRAIN #########
    # read in file
    im_path = in_path+meta.loc[i,"brain"]+"_brainreg_"+align_brain+"/downsampled_standard.tiff"

    im = tf.imread(im_path)
    # subtract mean value
    # im_sub = im - meta.loc[i, "mean_subtract"]

    # # sets negative numbers as +2^16, so set these as 0
    # im_sub[im_sub > 60000] = 0

    # # rotate 90 degrees so same orientation as atlas
    # im_rot = np.rot90(im_sub, k=-1, axes=(1,2))
    # # im_rot = np.rot90(im, k=-1, axes=(1,2))

    # # also flip left/right to get it in proper orientation
    # im_flip = np.flip(im_rot, axis=2)


    # save as norm_subtracted files
    print(out_path+meta.loc[i, 'brain']+"_aligned_to_"+align_brain+"_NO_subtracted.tif")
    # tf.imwrite(out_path+meta.loc[i, 'brain']+"_aligned_"+align_brain+"_norm_subtracted.tif", im_flip)
    # tf.imwrite(out_path+meta.loc[i, 'brain']+"_aligned_to_"+align_brain+"_norm_subtracted.tif", im_sub) 

    # for some reason doesn't work if imagej=True

    # save as NO subtraction
    tf.imwrite(out_path+meta.loc[i, 'brain']+"_aligned_to_"+align_brain+"_NO_subtracted.tif", im) 
    print("\n")
    


