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

in_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/input_tifs/norm_subtracted/STeg_220429_aligned/"
out_path = "/mnt/labNAS/Emily/STP_for_MAPseq/4_python_output/output_figs/"

metadata = pd.read_csv("stp_metadata.csv")

# set vid parameters
inj = "ACC"

# select inj site
inj_meta = metadata[metadata["inj_site"]==inj]

# select species
# steg
st_inj = inj_meta[inj_meta['species'] == "STeg"]
mm_inj = inj_meta[inj_meta['species'] == "MMus"]

# import images
st_paths = []
for i in list(st_inj.index):
    st_paths.append(in_path+metadata.loc[i,'brain']+"_aligned_to_STeg_220429_NO_subtracted.tif")
st_im = [tf.imread(path) for path in st_paths]
st_mean = np.mean(st_im, axis=0)

mm_paths = []
for i in list(mm_inj.index):
    mm_paths.append(in_path+metadata.loc[i,'brain']+"_aligned_to_STeg_220429_NO_subtracted.tif")
mm_im = [tf.imread(path) for path in mm_paths]
mm_mean = np.mean(mm_im, axis=0)

# start napari
viewer = napari.Viewer()

viewer.add_image(
    st_mean,
    name="steg_"+inj+"_mean",
    scale=[2.5,1,1],
    blending="additive",
    colormap="bop orange"
)

viewer.add_image(
    mm_mean,
    name="mmus_"+inj+"_mean",
    scale=[2.5,1,1],
    blending="additive",
    colormap="bop blue"
)
    
# add atlas boundaries
steg_220429_bound =  home_dir+"Emily/STP_for_MAPseq/3_brainreg_output/OMC_STeg_220429_b2_hand_straightened_asr_aligned_10um/boundaries_RESIZED.tif"
bound_im = tf.imread(steg_220429_bound)
# bound_flip = np.flip(bound_im, axis=2)
viewer.add_image(
    bound_im,
    name="boundaries",
    scale=[2.5,1,1],
    blending="additive",
    contrast_limits=[0,10]
)


animation = Animation(viewer)
viewer.update_console({'animation': animation})

viewer.dims.ndisplay = 3
viewer.camera.angles = (0.0, 180, 90)
animation.capture_keyframe()
# viewer.camera.zoom = 2.4
# animation.capture_keyframe()
viewer.camera.angles = (0, 360, 90)
animation.capture_keyframe(steps=60)
viewer.camera.angles = (0, 450, 90)
animation.capture_keyframe(steps=60)
viewer.camera.angles = (0, 540, 90)
animation.capture_keyframe(steps=60)
animation.animate(out_path+'lm_sm_3d_rot_'+inj+'.mov', canvas_only=True)