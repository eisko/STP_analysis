# load packages
import numpy as np
import pandas as pd
from skimage import io # import tiff file as ndarray
from STP_processing import make_mask
from time import time

start = time()

# for acadia!
in_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/"
out_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/masks/"

# load atlases
steg_reg_atlas = io.imread(in_path+"Steg_220429_registered_atlas_RESIZED.tif", plugin="tifffile")

mmus_reg_atlas = io.imread(in_path+"MMus_220303_registered_atlas_RESIZED.tif", plugin="tifffile")

atlases = [steg_reg_atlas, mmus_reg_atlas]
atlas_labels = ["STeg_220429", "MMus_220303"]

# areas to generate masks for
areas = ["grey", "CTX", "TH", "STR", "CP", "P", "MB", "PAG", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU"]


# loop through atlases
for i in range(len(atlases)):
    #loop through areas:
    for j in range(len(areas)):
        st_start = time()
        print("working on", atlas_labels[i], areas[j])
        area_mask = make_mask(areas[j], atlases[i])
        with open(out_path+atlas_labels[i]+"_"+areas[j]+".npy", "wb") as f:
            np.save(f, area_mask, allow_pickle=False)
        st_end = time()
        print(atlas_labels[i], areas[j], "took", st_end-st_start, "seconds")
        print("\n")
        
end = time()

print("script took", end-start, "seconds")