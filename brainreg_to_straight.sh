#!/bin/bash

# Align all brains to certain straightened brain

# make sure to `conda activate brainglobe` before running

straight_brain="STeg_220429"

# Replace 'your_csv_file.csv' with the actual filename of your CSV file
csv_file="/mnt/labNAS/Emily/STP_for_MAPseq/3_brainreg_output/brainreg_inpaths.csv"
out_folder="/mnt/labNAS/Emily/STP_for_MAPseq/3_brainreg_output/${straight_brain}_aligned"

# Check if the CSV file exists
if [ ! -f "$csv_file" ]; then
  echo "CSV file not found: $csv_file"
  exit 1
fi

# Check if out_folder already exists
if [ -d "$out_folder" ]; then
  echo "Folder '$out_folder' already exists."
else
  # Create the folder if it doesn't exist
  mkdir "$out_folder"
  echo "Folder '$out_folder' created successfully."
fi


# Read the CSV file line by line and print the first and second columns
while IFS= read -r line; do
  # Extract the first column (filename identifier)
  name=$(echo "$line" | cut -d ',' -f 1)

  # Extract the second column (file path)
  file_path=$(echo "$line" | cut -d ',' -f 2)

  # create subfolders for each aligned brain
  # Check if out_folder already exists
  if [ -d "${out_folder}/${name}_brainreg_${straight_brain}" ]; then
    echo "Folder '${name}_brainreg_${straight_brain}' already exists."
  else
    # Create the folder if it doesn't exist
    mkdir "${out_folder}/${name}_brainreg_${straight_brain}"
    echo "Folder '${name}_brainreg_${straight_brain}' created successfully."
  fi

  # run brainreg
  brainreg $file_path "${out_folder}/${name}_brainreg_${straight_brain}" \
  -v 50 20 20 --orientation asl --atlas allen_mouse_50um

done < $csv_file