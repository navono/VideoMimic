SHELL := /bin/bash

# --------------- Configurable variables ---------------
VIDEO_NAME  ?= my_video
STRIDE      ?= 2
ROBOT       ?= g1
HEIGHT      ?= -1
VIS         ?= 0
GENDER      ?= male

# --------------- Derived paths ---------------
BASE_DIR        := demo_data
CAM_DIR         := $(BASE_DIR)/input_images/$(VIDEO_NAME)/cam01
MASKS_DIR       := $(BASE_DIR)/input_masks/$(VIDEO_NAME)/cam01
POSE2D_DIR      := $(BASE_DIR)/input_2d_poses/$(VIDEO_NAME)/cam01
SMPL_DIR        := $(BASE_DIR)/input_3d_meshes/$(VIDEO_NAME)/cam01
CONTACT_DIR     := $(BASE_DIR)/input_contacts/$(VIDEO_NAME)/cam01
MEGASAM_DIR     := $(BASE_DIR)/input_megasam
MEGAHUNTER_DIR  := $(BASE_DIR)/output_smpl_and_points

# --------------- Auto-detect frame range ---------------
# Override by passing START_FRAME=xx END_FRAME=xx if needed
START_FRAME ?= $(shell ls $(CAM_DIR)/*.jpg 2>/dev/null | sort | head -1 | xargs -I{} basename {} .jpg)
END_FRAME   ?= $(shell ls $(CAM_DIR)/*.jpg 2>/dev/null | sort | tail -1 | xargs -I{} basename {} .jpg)

RESULT_TAG      := megasam_reconstruction_results_$(VIDEO_NAME)_cam01_frame_$(START_FRAME)_$(END_FRAME)_subsample_$(STRIDE)
MEGASAM_OUT     := $(MEGASAM_DIR)/$(RESULT_TAG).h5
MEGAHUNTER_OUT  := $(MEGAHUNTER_DIR)/megahunter_$(RESULT_TAG).h5
POSTPROC_DIR    := $(BASE_DIR)/output_calib_mesh/megahunter_$(RESULT_TAG)

# --------------- Conda helpers ---------------
CONDA_EVAL := eval "$$(conda shell.bash hook)"

# --------------- Targets ---------------

.PHONY: pipeline all _init_dirs preprocess reconstruct optimize postprocess retarget visualize extract-frames

# Create directories for the video
_init_dirs:
	@mkdir -p $(CAM_DIR) $(MASKS_DIR)/json_data $(POSE2D_DIR) $(SMPL_DIR) \
	          $(CONTACT_DIR) $(MEGASAM_DIR) $(MEGAHUNTER_DIR) $(POSTPROC_DIR)

# Full pipeline in a single shell (conda env switches work correctly)
pipeline: _init_dirs
	@$(CONDA_EVAL) && \
	echo "=== Step 0: Preprocessing ===" && \
	conda activate vm1rs && \
	bash preprocess_human.sh $(VIDEO_NAME) $(VIS) && \
	echo "=== Step 1: Reconstruction ===" && \
	conda activate vm1recon && \
	python stage1_reconstruction/megasam_reconstruction.py \
	    --video-dir $(CAM_DIR) --out-dir $(MEGASAM_DIR) \
	    --start-frame $(START_FRAME) --end-frame $(END_FRAME) \
	    --stride $(STRIDE) --gsam2 && \
	echo "=== Step 2: Optimization ===" && \
	conda activate vm1rs && \
	if [ "$(HEIGHT)" != "-1" ] && [ "$(HEIGHT)" != "0" ]; then \
	    python stage2_optimization/optimize_smpl_shape_for_height.py \
	        --height $(HEIGHT) --output-dir $(SMPL_DIR); \
	fi && \
	$$( [ "$(HEIGHT)" = "0" ] && [ "$(ROBOT)" = "g1" ] && echo "--use-g1-shape" || true ) > /tmp/_g1_flag && \
	python stage2_optimization/megahunter_optimization.py \
	    --world-env-path $(MEGASAM_OUT) \
	    --bbox-dir $(MASKS_DIR)/json_data \
	    --pose2d-dir $(POSE2D_DIR) \
	    --smpl-dir $(SMPL_DIR) \
	    --out-dir $(MEGAHUNTER_DIR) $$( [ "$(HEIGHT)" = "0" ] && [ "$(ROBOT)" = "g1" ] && echo "--use-g1-shape" || true ) && \
	echo "=== Step 3: Postprocessing ===" && \
	conda activate vm1recon && \
	python stage3_postprocessing/postprocessing_pipeline.py \
	    --megahunter-path $(MEGAHUNTER_OUT) \
	    --out-dir $(POSTPROC_DIR) \
	    --gender $(GENDER) --is-megasam && \
	echo "=== Step 4: Retargeting ===" && \
	conda activate vm1rs && \
	python stage4_retargeting/robot_motion_retargeting.py \
	    --src-dir $(POSTPROC_DIR) \
	    --contact-dir $(CONTACT_DIR) && \
	echo "Pipeline complete! Results in $(POSTPROC_DIR)"

# Kept for backward compat; prefer `make pipeline`
all: _init_dirs preprocess reconstruct optimize postprocess retarget
	@echo "Pipeline complete! Results in $(POSTPROC_DIR)"

# ---------- Step 0: Preprocessing (vm1rs) ----------
preprocess:
	$(CONDA_EVAL) && conda activate vm1rs && \
	bash preprocess_human.sh $(VIDEO_NAME) $(VIS)

# Individual preprocessing steps
sam2:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python stage0_preprocessing/sam2_segmentation.py \
	    --video-dir $(CAM_DIR) --output-dir $(MASKS_DIR)

vitpose:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python stage0_preprocessing/vitpose_2d_poses.py \
	    --video-dir $(CAM_DIR) --bbox-dir $(MASKS_DIR)/json_data \
	    --output-dir $(POSE2D_DIR)

vimo:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python stage0_preprocessing/vimo_3d_mesh.py \
	    --img-dir $(CAM_DIR) --mask-dir $(MASKS_DIR) \
	    --out-dir $(SMPL_DIR)

bstro:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python stage0_preprocessing/bstro_contact_detection.py \
	    --video-dir $(CAM_DIR) --bbox-dir $(MASKS_DIR)/json_data \
	    --output-dir $(CONTACT_DIR) \
	    --feet-contact-ratio-thr 0.2 --contact-thr 0.95

# ---------- Step 1: Reconstruction (vm1recon) ----------
reconstruct:
	$(CONDA_EVAL) && conda activate vm1recon && \
	python stage1_reconstruction/megasam_reconstruction.py \
	    --video-dir $(CAM_DIR) --out-dir $(MEGASAM_DIR) \
	    --start-frame $(START_FRAME) --end-frame $(END_FRAME) \
	    --stride $(STRIDE) --gsam2

# ---------- Step 2: Optimization (vm1rs) ----------
optimize:
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(call _optimize_body)

# Height-specific shape fitting (run before optimize if HEIGHT != -1 and HEIGHT != 0)
shape-fit:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python stage2_optimization/optimize_smpl_shape_for_height.py \
	    --height $(HEIGHT) --output-dir $(SMPL_DIR)

# ---------- Step 3: Postprocessing (vm1recon) ----------
postprocess:
	$(CONDA_EVAL) && conda activate vm1recon && \
	python stage3_postprocessing/postprocessing_pipeline.py \
	    --megahunter-path $(MEGAHUNTER_OUT) \
	    --out-dir $(POSTPROC_DIR) \
	    --gender $(GENDER) --is-megasam

# ---------- Step 4: Retargeting (vm1rs) ----------
retarget:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python stage4_retargeting/robot_motion_retargeting.py \
	    --src-dir $(POSTPROC_DIR) \
	    --contact-dir $(CONTACT_DIR)

# ---------- Visualization ----------
visualize:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python visualization/complete_results_egoview_visualization.py \
	    --postprocessed-dir $(POSTPROC_DIR) \
	    --robot-name $(ROBOT) --is-megasam

# ---------- Utility: extract frames ----------
extract-frames: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	python utilities/extract_frames_from_video.py \
	    --video-path $(VIDEO_PATH) \
	    --output-dir $(CAM_DIR)

# --------------- Internal ---------------
define _optimize_body
	$(eval _G1_FLAG := $(shell [ "$(HEIGHT)" = "0" ] && [ "$(ROBOT)" = "g1" ] && echo "--use-g1-shape"))
	if [ "$(HEIGHT)" != "-1" ] && [ "$(HEIGHT)" != "0" ]; then \
	    python stage2_optimization/optimize_smpl_shape_for_height.py \
	        --height $(HEIGHT) --output-dir $(SMPL_DIR); \
	fi && \
	python stage2_optimization/megahunter_optimization.py \
	    --world-env-path $(MEGASAM_OUT) \
	    --bbox-dir $(MASKS_DIR)/json_data \
	    --pose2d-dir $(POSE2D_DIR) \
	    --smpl-dir $(SMPL_DIR) \
	    --out-dir $(MEGAHUNTER_DIR) $(_G1_FLAG)
endef
