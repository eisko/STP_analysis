# Data Description
230721, ECI

This is a file to describe experimental details, folder structure, and processing pipeline in this folder.

**NOTE:** This folder does not contain python analysis scripts/notebooks. These are saved to my github, which you can find here: https://github.com/eisko/STP_analysis

# Experimental Details
- All brains were injected w/ AAV-Cre + AAV-Flox-TdT in either the OMC or ACC
- All brains were imaged on the STPT, while not all the same microscope, the same imaging settings were used.

There are 12 total brains:

| inj   | file  | STP microscope | Notes    |
|-------|-------|----------------|----------|
| 3 OMC MMus    | OMC_MMus_220119_b0 | Godzilla |   |
|     | OMC_MMus_220303_b1 | Godzilla | Used this brain for **alignment** and **visualization**|
|     | OMC_MMus_220324_b2 | Godzilla |  |
| 3 OMC STeg    | OMC_STeg_220208_b0 | Godzilla |    |
|     | OMC_STeg_220411_b1 | Godzilla | |
|     | OMC_STeg_220429_b2 | Herbie | Used this brain for STeg **alignments**, <br /> also used this brain for figure **visualization** |
| 3 ACC MMus    | ACC_MMus_230403_b1 | Marks |  |
|     | ACC_MMus_230404_b2 | Marks | Used this brain for for **visualization** |
|     | ACC_MMus_230427_b3 | Marks |    |
| 3 ACC STeg    | ACC_STeg_230308_b1 | Marks | Used this brain for fig 1 **visualization**, align w/ 220429 |
|     | ACC_STeg_230322_b2 | Marks |    |
|     | ACC_STeg_230501_b3 | Marks | Had trouble pre-processing this brain <br /> couldn't feed it through stitching pipeline |



## Transferring data
To transfer data from brain4/brainstore8 server to synology, I used scp. See example below:

```
# start screen session
screen

# navigate to folder containing raw data folder to transfer
scp -rv ./230501_EI_Steg_b3_AAV9_ACA_TdT \
banerjeelab@SINGINGMOUSE:/volume1/Data/Emily/STP_for_MAPseq/raw_data/

# to detach from screen session:
^a^d 
```

# 0. Raw Data
This folder contains data for __12__ brains. The data is the raw image data obtained from the microscope, i.e. unprocessed tiles. This data is what gets processed on brian4.

# 1. Processed Data
This folder contains the output of the processing pipeline. The pipline stitched together tiles from the same image and does some smoothing and image correction. It also produces a downsampled (warping) inmage.

The output for a single brain contains the following files/folders:
- **Log**
    - Mosaic files used for running pipeline
- **stitchedImage_ch1**
    - 1x1x50 um high-res images
- **stitchedImage_ch2**
- **warping**
    - 20x20x50 um downsampled image stack (...p05.tif)
- **averageTile_ch1.tif**
    - used for background subtraction?
- **averageTile_ch2.tif**
- **README.txt**
    - my notes on brain
- **runReconstruction_part1.sh**
    - bash script used to initiate processing pipeline

# Fiji Adjusted
Used **OMC_STeg_220429_b2** to align Steg brains. Used **OMC_MMus_220303_b1** to align MMus brains in next step (brainreg)

**NOTE**: did not use norm_substracted.tif for alignment, as this gets rid of background used to detect structure and align brains
Adjustments made on all brains = normalized background fluorescence:
- measure mean backgound in portion of brain w/o projections
    - used hippocampus b/c known no projection area
- subtract this mean from all pixels
- these files are named `brain_name_norm_subtracted.tif`

For brains used for alignment, picked mmus/steg that looked most ideally oriented. Tried to minimize amount of interpolation needed to prevent data loss.
- did initial rotation to get brains midline aligned to 90 degree
    - use line and angle tool in fiji
- Looked at reslice to try to improve A/P and M/L alignment, picked brains that didn't need alignment/interpolation
- Applied translation so brains roughly in center
- These files are named `brain_name_hand_straightened.tif`
- Also adjusted/flipped so that injection site is in correct orientation (asr)

|brain | mean subtract | Straightening/alignment |
|-----|---------|--------|
| OMC_MMus_220119_b0 | 299 |
| OMC_MMus_220303_b1 | 257 | Rotate: 4 <br /> Translate: x=-38, y=52 | 
| OMC_MMus_220324_b2 | 379.5 |
| OMC_STeg_220208_b0 | 309 |
| OMC_STeg_220411_b1 | 313.6 |
| OMC_STeg_220429_b2 | 308 | Rotate: 2 <br /> Translate: x=-5, y=1 |
| ACC_MMus_230403_b1 | 16
| ACC_MMus_230404_b2 | 20.6
| ACC_MMus_230427_b3 | ---
| ACC_STeg_230308_b1 | 21 |
| ACC_STeg_230322_b2 | 21 |
| ACC_STeg_230501_b3 | 19.7 |


# Brainreg output

## 1. align straightened brains to 10um atlas
First, align allen atlas to reference brains (OMC_MMus_220303 and OMC_STeg_220429). Aligned 10um atlas so that can size down atlas to right size w/o losing boundaries (rather than size up 25um atlas, chose to size down 10 um atlas).

NOTE: use p05 straightened w/o norm subtracted to get good fitting of atlas to brain

To align, used `brainglobe` conda environment. Note: took ~1 hour to align one brain to 10um atlas
```
# to align MMus_220303 to allen_10um
brainreg /mnt/labNAS/Emily/STP_for_MAPseq/2_fiji_adjusted/OMC_MMus_220303_b1_norm_subtracted_hand_straightened.tif \
/mnt/labNAS/Emily/STP_for_MAPseq/3_brainreg_output/OMC_MMus_220303_normaligned_10um_brainreg \
-v 50 20 20 \
--orientation asl \
--atlas allen_mouse_10um \

# to align STeg_220429 to allen_10um
brainreg /mnt/labNAS/Emily/STP_for_MAPseq/2_fiji_adjusted/OMC_STeg_220429_b2_hand_straightened.tif \
/mnt/labNAS/Emily/STP_for_MAPseq/3_brainreg_output/OMC_STeg_220429_normaligned_10um_brainreg \
-v 50 20 20 \
--orientation asl \
--atlas allen_mouse_10um \
```

Next, I downsampled (rescaled) the aligned atlas in fiji to match dimensions of original data in fiji
- Image -> adjust -> size
    - unchecked averageing, no interpolation
- saved as `...resized_atlas_to_Steg.tif`

## 2. Register all brains to straightened brains

Next, aligned all brains to MMus_220303 and all brains to STeg_220429. 
- Did this by replacing allen_mouse_50um_v1.2 atlas w/ raw MMus/STeg straightened data and modifying the README file. 
    - replaced `reference.tiff` with `brain_name_hand_straightend_asr.tif` of mmus/steg
    - replaced `annotation.tiff` with `registered_atlas_RESIZED.tif` of mmus/steg
    - replaced `hemispheres.tiff` with `hemispheres_RESIZED.tif` of mmus/steg
    - modified `metadata.json`, eg.:
```
# for steg
{"name": "steg_fixed_220429", "citation": "OMC_STeg_220429_b2_hand_straightened_asr.tif", "atlas_link": "http://www.brain-map.org", "species": "Scotinomys teguina", "symmetric": false, "resolution": [50.0, 20.0, 20.0], "orientation": "asr", "version": "1.2", "shape": [201, 522, 692], "additional_references": []}

# for mmus
{"name": "mmus_fixed_220303", "citation": "OMC_MMus_220303_b2_hand_straightened_asr.tif", "atlas_link": "http://www.brain-map.org", "species": "Mus musculus", "symmetric": false, "resolution": [50.0, 20.0, 20.0], "orientation": "asr", "version": "1.2", "shape": [212, 554, 682], "additional_references": []}
```

- Created bash file `brainreg_to_straight.sh` to automatically align all brains to one reference brain after modifying allen_mouse_50um
    - takes about 40-50 mins to run on all 11 brains
- Learned need atlas link in metadata for program to run
- NOTE: Needed to use original p05/downsampled images **w/o** norm-subtraction for better alignment

# 4. Python Analysis
- Made script to automatically subtract mean background given path of files
    - takes `aligned_file_paths.csv` as input, find in 3_brainreg folder
    - saves `...norm_subtracted.tif` as output in `4_python/input` foler

## To compare/average replicates
- Find threshold to distinguish injection site
    1. Denoise image by guassian blur
    2. Auto-detect threshold - IsoData or Otsu method worked in fiji
    3. apply threshold to brain
    4. keep slices where injection site is as injection site mask
- need to normalize between brains
