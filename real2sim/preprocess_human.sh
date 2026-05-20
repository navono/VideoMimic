#!/bin/bash

# Usage: ./preprocess_human.sh <name> [<vis_flag; 1 or 0>] 
# Ex) vis: bash preprocess_human.sh fourleghuman_tutorial1_subset 1 
# Ex) no vis: bash preprocess_human.sh fourleghuman_tutorial1_subset 0 


# Input arguments
NAME=$1

# Strip path and extension, e.g. assets/sitting_standing.mp4 -> sitting_standing
VID_STEM=$(basename "${NAME%.*}")
NAME="$VID_STEM"

# Determine if --vis should be included
VIS_FLAG=""
if [ "$2" == "1" ]; then
    VIS_FLAG="--vis"
fi

# All data under demo_data/VID_STEM/
BASE_DIR="demo_data/$NAME"
CAM_DIR="$BASE_DIR/input_images/cam01"
MASKS_DIR="$BASE_DIR/input_masks/cam01"
POSE2D_DIR="$BASE_DIR/input_2d_poses/cam01"
SMPL_DIR="$BASE_DIR/input_3d_meshes/cam01"
CONTACT_DIR="$BASE_DIR/input_contacts/cam01"

mkdir -p "$MASKS_DIR/json_data" "$POSE2D_DIR" "$SMPL_DIR" "$CONTACT_DIR"

echo "Running Grounding-SAM-2..."
# CMD="python stage0_preprocessing/sam2_segmentation.py --video-dir \"$CAM_DIR\" --output-dir \"$MASKS_DIR\" $VIS_FLAG"
CMD="python stage0_preprocessing/sam2_segmentation.py --video-dir \"$CAM_DIR\" --output-dir \"$MASKS_DIR\" --vis"
echo "$CMD"
eval "$CMD"

echo -e "\nRunning ViTPose..."
CMD="python stage0_preprocessing/vitpose_2d_poses.py --video-dir \"$CAM_DIR\" --bbox-dir \"$MASKS_DIR/json_data\" --output-dir \"$POSE2D_DIR\" $VIS_FLAG"
echo "$CMD"
eval "$CMD"

echo -e "\nRunning VIMO..."
CMD="python stage0_preprocessing/vimo_3d_mesh.py --img-dir \"$CAM_DIR\" --mask-dir \"$MASKS_DIR\" --out-dir \"$SMPL_DIR\""
echo "$CMD"
eval "$CMD"

echo -e "\nRunning BSTRO..."
CMD="python stage0_preprocessing/bstro_contact_detection.py --video-dir \"$CAM_DIR\" --bbox-dir \"$MASKS_DIR/json_data\" --output-dir \"$CONTACT_DIR\"  --feet-contact-ratio-thr 0.2 --contact-thr 0.95"
echo "$CMD"
eval "$CMD"

echo -e "\nDone!"
