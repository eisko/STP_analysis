# load packages
import numpy as np
import pandas as pd
import napari
from napari_animation import Animation
import tifffile as tf
from skimage import io # import tiff file as ndarray
import os
import matplotlib.pyplot as plt

# for acadia
in_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/norm_subtracted/STeg_220429_aligned/"
out_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/output_figs/"

metadata = pd.read_csv("stp_metadata.csv")

# get file names
dir_list = os.listdir(in_path)

# set vid parameters
species = "STeg"
inj = "OMC"

# select species
sp_idx = metadata['species'] == species
sp_meta = metadata[sp_idx]

# select inj site
sp_inj_meta = sp_meta[sp_meta["inj_site"]==inj]
sp_inj_idx = list(sp_inj_meta.index)

# import images
images = []
for i in sp_inj_idx:
    images.append(tf.imread(in_path+dir_list[i]))

# start napari
viewer = napari.Viewer()

colors = ["green", "cyan", "magenta"]
for i in range(len(images)):
    viewer.add_image(
        images[i],
        name=sp_inj_meta.iloc[i, 0],
        scale=[2.5,1,1],
         # contrast_limits=[0,1],
         blending="additive",
        colormap=colors[i]
        )
    
# collect animation
animation = Animation(viewer)

for i in range(200):
    viewer.dims.set_point(0, i)
    animation.capture_keyframe()

animation.animate(out_path+species+"_"+inj+"_all"+".mp4")