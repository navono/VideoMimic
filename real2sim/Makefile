SHELL := /bin/bash

# --------------- Configurable variables ---------------
VIDEO_NAME  ?= my_video
STRIDE      ?= 2
ROBOT       ?= g1
HEIGHT      ?= -1
VIS         ?= 0
GENDER      ?= male

# --------------- Derived paths ---------------
# Strip path and extension from VIDEO_NAME, e.g. assets/sitting_standing.mp4 -> sitting_standing
VID_STEM        := $(basename $(notdir $(VIDEO_NAME)))
BASE_DIR        := demo_data/$(VID_STEM)
CAM_DIR         := $(BASE_DIR)/input_images/cam01
MASKS_DIR       := $(BASE_DIR)/input_masks/cam01
POSE2D_DIR      := $(BASE_DIR)/input_2d_poses/cam01
SMPL_DIR        := $(BASE_DIR)/input_3d_meshes/cam01
CONTACT_DIR     := $(BASE_DIR)/input_contacts/cam01
MEGASAM_DIR     := $(BASE_DIR)/input_megasam
MEGAHUNTER_DIR  := $(BASE_DIR)/output_smpl_and_points

# --------------- Auto-detect frame range ---------------
# Override by passing START_FRAME=xx END_FRAME=xx if needed
START_FRAME ?= $(shell ls $(CAM_DIR)/*.jpg 2>/dev/null | sort | head -1 | xargs -I{} basename {} .jpg)
END_FRAME   ?= $(shell ls $(CAM_DIR)/*.jpg 2>/dev/null | sort | tail -1 | xargs -I{} basename {} .jpg)

RESULT_TAG      := megasam_reconstruction_results_$(VID_STEM)_cam01_frame_$(START_FRAME)_$(END_FRAME)_subsample_$(STRIDE)
MEGASAM_OUT     := $(MEGASAM_DIR)/$(RESULT_TAG).h5
MEGAHUNTER_OUT  := $(MEGAHUNTER_DIR)/megahunter_$(RESULT_TAG).h5
POSTPROC_DIR    := $(BASE_DIR)/output_calib_mesh/megahunter_$(RESULT_TAG)

# --------------- Conda helpers ---------------
CONDA_EVAL := eval "$$(conda shell.bash hook)"

# --------------- Proxy & HF (override PROXY= to disable) ---------------
PROXY ?= http://127.0.0.1:18899
HF_TOKEN ?= xx
PROXY_ENV := $(if $(PROXY),http_proxy=$(PROXY) https_proxy=$(PROXY),) HF_TOKEN=$(HF_TOKEN)

# --------------- Targets ---------------

.PHONY: pipeline all _init_dirs preprocess reconstruct optimize postprocess retarget visualize extract-frames

# Create directories for the video
_init_dirs:
	@mkdir -p $(CAM_DIR) $(MASKS_DIR)/json_data $(POSE2D_DIR) $(SMPL_DIR) \
	          $(CONTACT_DIR) $(MEGASAM_DIR) $(MEGAHUNTER_DIR) $(POSTPROC_DIR)

# Full pipeline in a single shell (conda env switches work correctly)
pipeline: _init_dirs
	@$(CONDA_EVAL) && \
	$(if $(PROXY),export http_proxy=$(PROXY) https_proxy=$(PROXY) HF_TOKEN=$(HF_TOKEN) &&,) \
	$(if $(PROXY),,export HF_TOKEN=$(HF_TOKEN) &&) \
	echo "=== Step 0: Preprocessing ===" && \
	conda activate vm1rs && \
	bash preprocess_human.sh $(VIDEO_NAME) $(VIS) && \
	echo "=== Step 1: Reconstruction ===" && \
	conda activate vm1recon && \
	python stage1_reconstruction/megasam_reconstruction.py \
	    --video-dir $(CAM_DIR) --out-dir $(MEGASAM_DIR) \
	    --start-frame $(START_FRAME) --end-frame $(END_FRAME) \
	    --stride $(STRIDE) --gsam2 --localhub && \
	MEGASAM_OUT=$$(ls -t $(MEGASAM_DIR)/*.h5 2>/dev/null | head -1) && \
	echo "MegaSAM output: $$MEGASAM_OUT" && \
	echo "=== Step 2: Optimization ===" && \
	conda activate vm1rs && \
	if [ "$(HEIGHT)" != "-1" ] && [ "$(HEIGHT)" != "0" ]; then \
	    python stage2_optimization/optimize_smpl_shape_for_height.py \
	        --height $(HEIGHT) --output-dir $(SMPL_DIR); \
	fi && \
	python stage2_optimization/megahunter_optimization.py \
	    --world-env-path "$$MEGASAM_OUT" \
	    --bbox-dir $(MASKS_DIR)/json_data \
	    --pose2d-dir $(POSE2D_DIR) \
	    --smpl-dir $(SMPL_DIR) \
	    --out-dir $(MEGAHUNTER_DIR) $$( [ "$(HEIGHT)" = "0" ] && [ "$(ROBOT)" = "g1" ] && echo "--use-g1-shape" || true ) && \
	MEGAHUNTER_OUT=$$(ls -t $(MEGAHUNTER_DIR)/megahunter_*.h5 2>/dev/null | head -1) && \
	echo "MegaHunter output: $$MEGAHUNTER_OUT" && \
	RESULT_TAG=$$(basename "$$MEGAHUNTER_OUT" .h5 | sed 's/^megahunter_//') && \
	POSTPROC_DIR="$(BASE_DIR)/output_calib_mesh/$$RESULT_TAG" && \
	echo "=== Step 3: Postprocessing ===" && \
	conda activate vm1recon && \
	python stage3_postprocessing/postprocessing_pipeline.py \
	    --megahunter-path "$$MEGAHUNTER_OUT" \
	    --out-dir "$$POSTPROC_DIR" \
	    --gender $(GENDER) --is-megasam && \
	echo "=== Step 4: Retargeting ===" && \
	conda activate vm1rs && \
	python stage4_retargeting/robot_motion_retargeting.py \
	    --src-dir "$$POSTPROC_DIR" \
	    --contact-dir $(CONTACT_DIR) && \
	echo "Pipeline complete! Results in $$POSTPROC_DIR"

# Kept for backward compat; prefer `make pipeline`
all: _init_dirs preprocess reconstruct optimize postprocess retarget
	@echo "Pipeline complete! Results in $(POSTPROC_DIR)"

# ---------- Step 0: Preprocessing (vm1rs) ----------
preprocess: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) bash preprocess_human.sh $(VIDEO_NAME) $(VIS)

# Individual preprocessing steps
sam2: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python stage0_preprocessing/sam2_segmentation.py \
	    --video-dir $(CAM_DIR) --output-dir $(MASKS_DIR)

vitpose: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python stage0_preprocessing/vitpose_2d_poses.py \
	    --video-dir $(CAM_DIR) --bbox-dir $(MASKS_DIR)/json_data \
	    --output-dir $(POSE2D_DIR)

vimo: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python stage0_preprocessing/vimo_3d_mesh.py \
	    --img-dir $(CAM_DIR) --mask-dir $(MASKS_DIR) \
	    --out-dir $(SMPL_DIR)

bstro: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python stage0_preprocessing/bstro_contact_detection.py \
	    --video-dir $(CAM_DIR) --bbox-dir $(MASKS_DIR)/json_data \
	    --output-dir $(CONTACT_DIR) \
	    --feet-contact-ratio-thr 0.2 --contact-thr 0.95

# ---------- Step 1: Reconstruction (vm1recon) ----------
reconstruct: _init_dirs
	$(CONDA_EVAL) && conda activate vm1recon && \
	$(PROXY_ENV) python stage1_reconstruction/megasam_reconstruction.py \
	    --video-dir $(CAM_DIR) --out-dir $(MEGASAM_DIR) \
	    --start-frame $(START_FRAME) --end-frame $(END_FRAME) \
	    --stride $(STRIDE) --gsam2 --localhub

# ---------- Step 2: Optimization (vm1rs) ----------
optimize: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) $(call _optimize_body)

# Height-specific shape fitting (run before optimize if HEIGHT != -1 and HEIGHT != 0)
shape-fit: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python stage2_optimization/optimize_smpl_shape_for_height.py \
	    --height $(HEIGHT) --output-dir $(SMPL_DIR)

# ---------- Step 3: Postprocessing (vm1recon) ----------
postprocess: _init_dirs
	$(CONDA_EVAL) && conda activate vm1recon && \
	$(PROXY_ENV) python stage3_postprocessing/postprocessing_pipeline.py \
	    --megahunter-path $(MEGAHUNTER_OUT) \
	    --out-dir $(POSTPROC_DIR) \
	    --gender $(GENDER) --is-megasam

# ---------- Step 4: Retargeting (vm1rs) ----------
retarget: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python stage4_retargeting/robot_motion_retargeting.py \
	    --src-dir $(POSTPROC_DIR) \
	    --contact-dir $(CONTACT_DIR)

# ---------- Visualization ----------
visualize:
	@POSTPROC_DIR=$$(for d in $(BASE_DIR)/output_calib_mesh/*/; do [ -f "$$d/retarget_poses_g1.h5" ] && echo "$$d" && break; done) && \
	if [ -z "$$POSTPROC_DIR" ]; then echo "Error: no complete results found in $(BASE_DIR)/output_calib_mesh/"; exit 1; fi && \
	echo "Visualizing: $$POSTPROC_DIR" && \
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python visualization/complete_results_egoview_visualization.py \
	    --postprocessed-dir "$$POSTPROC_DIR" \
	    --robot-name $(ROBOT) --is-megasam

# ---------- Utility: extract frames ----------
# Auto-detect total frames via ffprobe if END_FRAME not set
TOTAL_FRAMES   := $(shell ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $(VIDEO_PATH) 2>/dev/null)
EXTRACT_START  ?= 0
EXTRACT_END    ?= $(TOTAL_FRAMES)
extract-frames: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) python utilities/extract_frames_from_video.py \
	    --video-path $(VIDEO_PATH) \
	    --output-dir $(CAM_DIR) \
	    --start-frame $(EXTRACT_START) --end-frame $(EXTRACT_END)

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

# --------------- HTTP API Server ---------------
SERVER_PORT ?= 8090
SERVER_LOG  ?= /tmp/videomimic-server.log

.PHONY: serve serve-install serve-bg serve-stop serve-log

# Install server dependencies
serve-install:
	$(CONDA_EVAL) && conda activate vm1rs && \
	pip install -r server/server_requirements.txt

# Foreground (for development / debugging)
serve:
	$(CONDA_EVAL) && conda activate vm1rs && \
	python server/server.py

# Background (for production / persistent)
serve-bg:
	$(CONDA_EVAL) && conda activate vm1rs && \
	nohup python server/server.py > $(SERVER_LOG) 2>&1 & echo "PID=$$!"

# Stop background server
serve-stop:
	@pkill -f 'python server/server.py' 2>/dev/null && echo 'Server stopped' || echo 'No server running'

# Tail server log
serve-log:
	@tail -f $(SERVER_LOG) 2>/dev/null || echo 'No log file at $(SERVER_LOG)'
