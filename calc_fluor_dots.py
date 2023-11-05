# load packages
import numpy as np
import pandas as pd
# import napari
import tifffile as tf
from skimage import io # import tiff file as ndarray
import os
import matplotlib.pyplot as plt

# import custum colormaps
from colormaps import *

# import custum functions
from STP_plotting import *
from STP_processing import *


# choose based on run in acadia or home computer
home_dir = "/mnt/labNAS/"
# home_dir = "/Volumes/Data/"

metadata = pd.read_csv("stp_metadata.csv")

in_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/input_tifs/"
out_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/output_figs/"

# add paths to metadata
# output of mean_subtract_individuals.py
for i in range(metadata.shape[0]):
    b_path = in_path+"norm_subtracted/p05_mean_subtracted/"+metadata.loc[i, 'brain']+"_p05_NO_subtracted.tif"

    metadata.loc[i,"p05_path"] = b_path

# area list from make_masks.py
areas = ["OMCi", "OMCc", "aud","TH", "STR", "CP", "AMY", "HY","P", "PG", "PAG", "SCm", 
         "SNr", "BS"]

# load masks for each individually aligned brain
from time import time
start = time()
# make dictionary where keys = areas, values=list of masks corresponding to metadata order
mask_dict = {}
for area in areas:
    area_masks = []
    print("working on", area)
    for i in range(metadata.shape[0]):
        print("\t", metadata.loc[i,"brain"])
        save_folder = in_path+"masks/"+metadata.loc[i,"brain"]+"_masks/"
        with open(save_folder+metadata.loc[i,"brain"]+"_"+area+".npy", "rb") as f:
            area_masks.append(np.load(f))
        
    mask_dict[area] = area_masks

# for area in mask_dict:
#     print(area, ": \t", len(mask_dict[area]))

end = time()
print("took", end-start, "seconds to load")
# import images
# import images
p05_path = in_path+"norm_subtracted/p05_mean_subtracted/"
p05_images = []
for i in range(metadata.shape[0]):
    print("loading:", metadata.loc[i, "brain"])
    p05_images.append(tf.imread(p05_path+metadata.loc[i,'brain']+"_p05_NO_subtracted.tif"))

# set areas to focus on
areas_plot = ["OMCc", "aud", "CP", "AMY", "TH", "HY", 
              "SNr", "SCm", "PG", "PAG", "BS"]
all_fluor = calc_fluor_individ(p05_images, metadata, mask_dict, 
                               areas_to_plot=areas_plot)

# calculate fluor/vol
# scatterplot
omc_fluor = all_fluor[all_fluor["inj_site"]=="OMC"]
omc_fluor["Fluor/Vol"] = omc_fluor["Fluorescence"]/omc_fluor["Volume_mm3"]

omc_fluor.to_csv("OMC_fl_vol.csv", index=False)