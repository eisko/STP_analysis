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
    b_path = in_path+"norm_subtracted/p05_mean_subtracted/"+metadata.loc[i, 'brain']+"_p05_norm_subtracted.tif"

    metadata.loc[i,"path"] = b_path

# area list from make_masks.py
areas = ["grey", "CTX", "OMCc", "ACAc", "aud","TH", "STR", "CP", "AMY", "P", "PG", "MB", "PAG", "SCm", 
         "SNr", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU", "BS"]

# load images
images = []
for i in range(metadata.shape[0]):
    print('loading:', metadata.loc[i,"path"])
    images.append(tf.imread(metadata.loc[i,"path"]))

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
# quantify integrated fluorescence



## OMC
for i in range(metadata.shape[0]):
    print(metadata.loc[i,"brain"])
    print("image shape:", images[i].shape)
    print("mask shape: ", mask_dict['OMCc'][i].shape)
    print("\n")
    # print(images[i].shape)

# set areas that match w/ mapseq data
areas_plot = ["OMCc", "CP", "aud", "AMY", "TH", "HY", "SNr", "SCm", "PG", "PAG", "BS"]

omc_fluor = calc_fluor_individ(images, metadata, mask_dict, areas_to_plot=areas_plot, inj_site="OMC")
# save omc_fluor
omc_fluor.to_csv(out_path+"omc_fluor")

# make dot plot
plt.figure()
dot_bar_plot(omc_fluor, title="OMC", xaxis="area", yaxis="Fluorescence", hueaxis="species")
plt.savefig(out_path+"OMC_fluor_dotplot.jpeg", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
stvmm_area_scatter(omc_fluor, title="OMC Injections")
plt.savefig(out_path+"OMC_fluor_scatter.jpeg", dpi=300, bbox_inches="tight")
plt.close()

## ACC
areas_plot = ["ACAc", "CP", "aud", "AMY", "TH", "HY", "SNr", "SCm", "PG", "PAG", "BS"]

acc_fluor = calc_fluor_individ(images, metadata, mask_dict, areas_to_plot=areas_plot, inj_site="ACC")
acc_fluor.to_csv(out_path+"acc_fluor")

# make dot plot
plt.figure()
dot_bar_plot(acc_fluor, title="ACC", xaxis="area", yaxis="Fluorescence", hueaxis="species")
plt.savefig(out_path+"ACC_fluor_dotplot.jpeg", dpi=300, bbox_inches="tight")
plt.close()

# make scatterplot
plt.figure()
stvmm_area_scatter(acc_fluor, title="ACC Injections")
plt.savefig(out_path+"ACC_fluor_scatter.jpeg", dpi=300, bbox_inches="tight")
plt.close()

