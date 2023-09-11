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
home_dir = "/mnt/labNAS/"

in_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/input_tifs/norm_subtracted/STeg_220429_aligned/"
out_path = home_dir+"Emily/STP_for_MAPseq/4_python_output/output_figs/"

metadata = pd.read_csv("stp_metadata.csv")

# import images
steg_path = in_path+metadata.loc[4,'brain']+"_aligned_to_STeg_220429_NO_subtracted.tif"
mmus_path = in_path+metadata.loc[1,'brain']+"_aligned_to_STeg_220429_NO_subtracted.tif"

# atlas boundaries
steg_220429_bound =  home_dir+"Emily/STP_for_MAPseq/4_python_output/input_tifs/resized_atlases/allen_10um_to_OMC_STeg_220429_b2_boundaires_RESIZED.tiff"


steg_im = tf.imread(steg_path)
mmus_im = tf.imread(mmus_path)
bound_im = tf.imread(steg_220429_bound)

# start napari
viewer = napari.Viewer()

viewer.add_image(
    steg_im,
    name=metadata.loc[4,'brain'],
    scale=[2.5,1,1],
    blending="additive",
    colormap="bop orange",
    contrast_limits=[0,5000]
)

viewer.add_image(
    mmus_im,
    name=metadata.loc[1,'brain'],
    scale=[2.5,1,1],
    blending="additive",
    colormap="bop blue",
    contrast_limits=[0,5000]
)

# viewer.add_image(
#     bound_im,
#     name="boundaries",
#     scale=[2.5,1,1],
#     blending="additive",
#     opacity=0.2
# )

bound_flip = np.flip(bound_im, axis=2)
viewer.add_image(
    bound_flip,
    name="boundaries_flipped",
    scale=[2.5,1,1],
    blending="additive",
    opacity=0.2
)
    
# collect animation
animation = Animation(viewer)

for i in range(25):
    viewer.dims.set_point(0, i*20)
    animation.capture_keyframe()

animation.animate(out_path+metadata.loc[4,'brain']+"_"+metadata.loc[1,'brain']+"_steg_220429_bound_NO_sub"+".mp4")