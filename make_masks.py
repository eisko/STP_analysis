# load packages
import numpy as np
import pandas as pd
from skimage import io # import tiff file as ndarray
from STP_processing import make_mask
from time import time

start = time()



# for acadia!
home_dir="/mnt/labNAS/"
in_path = "/mnt/labNAS/Emily/STP_for_MAPseq/3_brainreg_output/OMC_STeg_220429_b2_hand_straightened_asr_aligned_10um/"
out_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/masks/steg_220429_hs_asr/"

atlas_label = "steg_220429_hs_asr"

# load atlases
steg_reg_atlas = io.imread(in_path+"registered_atlas_RESIZED.tif", plugin="tifffile")
# mmus_reg_atlas = io.imread(in_path+"MMus_220303_registered_atlas_RESIZED.tif", plugin="tifffile")

# load hemispheres
steg_reg_hemi = io.imread(in_path+"registered_hemispheres_RESIZED.tif", plugin="tifffile")
# mmus_reg_hemi = io.imread(in_path+"MMus_220303_registered_hemispheres_RESIZED.tif", plugin="tifffile")


# atlases = [steg_reg_atlas, mmus_reg_atlas]
# hemis = [steg_reg_hemi, mmus_reg_hemi]
# atlas_label = ["STeg_220429", "MMus_220303"]

# areas to generate masks for
areas = ["grey", "CTX", "OMCc", "ACAc", "aud","TH", "STR", "CP", "AMY", "P", "PG", "MB", "PAG", "SCm", 
         "SNr", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU", "BS", "HIP"]


# loop through atlases
# for i in range(len(atlases)):
#loop through areas:
for j in range(len(areas)):
    st_start = time()
    print("working on", atlas_label, areas[j])

    # define special cases
    if areas[j] == "OMCc":
        mos = make_mask("MOs", steg_reg_atlas)
        mop = make_mask("MOp", steg_reg_atlas)
        left_hemi = steg_reg_hemi==1
        omc = np.add(mos,mop)
        omcc = np.multiply(omc, left_hemi)
        omcc[omcc>0] = 1
        area_mask = omcc
    elif areas[j] == "ACAc":
        aca = make_mask("ACA", steg_reg_atlas)
        left_hemi = steg_reg_hemi==1
        acac = np.multiply(aca, left_hemi)
        acac[acac>0] = 1
        area_mask = acac
    elif areas[j] == "aud":
        tea = make_mask("TEa", steg_reg_atlas)
        visc = make_mask("VISC", steg_reg_atlas)
        ect = make_mask("ECT", steg_reg_atlas)
        tea_visc = np.add(tea, visc)
        aud = np.add(tea_visc, ect)
        aud[aud>0] = 1
        area_mask = aud
    elif areas[j] == "AMY":
        bma = make_mask("BMA", steg_reg_atlas)
        bla = make_mask("BLA", steg_reg_atlas)
        la = make_mask("LA", steg_reg_atlas)
        bma_bla = np.add(bma, bla)
        amy = np.add(bma_bla, la)
        amy[amy>0] = 1
        area_mask = amy
    elif areas[j] == "HY":
        hy = make_mask("HY", steg_reg_atlas)
        zi = make_mask("ZI", steg_reg_atlas)
        hy = np.subtract(hy, zi)
        hy[hy<1] = 0
        area_mask=hy
    elif areas[j] == "BS":
        grn = make_mask("GRN", steg_reg_atlas)
        irn = make_mask("IRN", steg_reg_atlas)
        bs = np.add(grn,irn)
        bs[bs>0] = 1
        area_mask = bs
    else:
        area_mask = make_mask(areas[j], steg_reg_atlas)


    # convert area_mask type to boolean to reduce size
    area_mask = area_mask.astype("bool")

    with open(out_path+atlas_label+"_"+areas[j]+".npy", "wb") as f:
        np.save(f, area_mask, allow_pickle=False)


    st_end = time()
    print(atlas_label, areas[j], "took", st_end-st_start, "seconds")
    print("\n")
        

end = time()


print("script took", end-start, "seconds")