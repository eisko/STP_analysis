# make masks for each atlas aligned to individual brain

# load packages
import os
import numpy as np
import pandas as pd
import tifffile as tf # import tiff file as ndarray
from STP_processing import make_mask
from time import time


def create_output_folder(output_directory, folder_name):
    folder_path = os.path.join(output_directory, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created in '{output_directory}'.")
    else:
        print(f"Folder '{folder_name}' already exists in '{output_directory}'.")


#######################
start = time()

# import metadata
metadata = pd.read_csv("stp_metadata.csv")

# for acadia!
in_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/resized_atlases/"
out_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/masks/"

# areas to generate masks for
areas = ["grey", "CTX", "OMCc", "ACAc", "aud","TH", "STR", "CP", "AMY", "P", "PG", "MB", "PAG", "SCm", 
         "SNr", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU", "BS"]

# create masks for every brain
for i in range(metadata.shape[0]):
    create_output_folder(out_path, metadata.loc[i,"brain"]+"_masks")

# load atlases
atlases = []
hemis = []
print("importing registered atlases and hemis")
for i in range(metadata.shape[0]):
    atlas_path = in_path+"allen_10um_to_"+metadata.loc[i,"brain"]+"_atlas_RESIZED.tiff"
    hemi_path = in_path+"allen_10um_to_"+metadata.loc[i,"brain"]+"_hemis_RESIZED.tiff"
    atlases.append(tf.imread(atlas_path))
    hemis.append(tf.imread(hemi_path))

# loop through atlases
for i in range(len(atlases)):
    #loop through areas:
    for j in range(len(areas)):
        st_start = time()
        print("working on", metadata.loc[i,"brain"], areas[j])

        # check if mask already created -> only run loop if does not exist
        mask_path = out_path+metadata.loc[i,"brain"]+"_masks/"+metadata.loc[i,"brain"]+"_"+areas[j]+".npy"
        if os.path.exists(mask_path):
            print(metadata.loc[i,"brain"]+"_"+areas[j]+".npy", "already exists")
        else:
            
            # define special cases
            if areas[j] == "OMCc":
                mos = make_mask("MOs", atlases[i])
                mop = make_mask("MOp", atlases[i])
                left_hemi = hemis[i]==2
                omc = np.add(mos,mop)
                omcc = np.multiply(omc, left_hemi)
                omcc[omcc>0] = 1
                area_mask = omcc
            elif areas[j] == "ACAc":
                aca = make_mask("ACA", atlases[i])
                left_hemi = hemis[i]==2
                acac = np.multiply(aca, left_hemi)
                acac[acac>0] = 1
                area_mask = acac
            elif areas[j] == "aud":
                tea = make_mask("TEa", atlases[i])
                visc = make_mask("VISC", atlases[i])
                ect = make_mask("ECT", atlases[i])
                tea_visc = np.add(tea, visc)
                aud = np.add(tea_visc, ect)
                aud[aud>0] = 1
                area_mask = aud
            elif areas[j] == "AMY":
                bma = make_mask("BMA", atlases[i])
                bla = make_mask("BLA", atlases[i])
                la = make_mask("LA", atlases[i])
                bma_bla = np.add(bma, bla)
                amy = np.add(bma_bla, la)
                amy[amy>0] = 1
                area_mask = amy
            elif areas[j] == "HY":
                hy = make_mask("HY", atlases[i])
                zi = make_mask("ZI", atlases[i])
                hy = np.subtract(hy, zi)
                hy[hy<1] = 0
                area_mask=hy
            elif areas[j] == "BS":
                grn = make_mask("GRN", atlases[i])
                irn = make_mask("IRN", atlases[i])
                bs = np.add(grn,irn)
                bs[bs>0] = 1
                area_mask = bs
            else:
                area_mask = make_mask(areas[j], atlases[i])


            # convert area_mask type to boolean to reduce size
            area_mask = area_mask.astype("bool")

            save_folder = out_path+metadata.loc[i,"brain"]+"_masks/"
            with open(save_folder+metadata.loc[i,"brain"]+"_"+areas[j]+".npy", "wb") as f:
                np.save(f, area_mask, allow_pickle=False)


        st_end = time()
        print(metadata.loc[i,"brain"], areas[j], "took", st_end-st_start, "seconds")
        print("\n")
        

end = time()


print("script took", end-start, "seconds")
