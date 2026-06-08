SHELL := /bin/bash

# --------------- Configurable variables ---------------
# VIDEO_PATH: path to the video file (e.g. assets/sitting_standing.mp4)
VIDEO_PATH  ?= ./assets/sitting_standing.mp4
STRIDE      ?= 2
ROBOT       ?= g1
HEIGHT      ?= -1
VIS         ?= 0
GENDER      ?= male

# --------------- Convert to AdaMimic variables ---------------
# TASK_NAME:  name of the motion task (used as AdaMimic dataset folder name)
# CONVERT_INPUT:  path to retarget_poses_g1.h5 (auto-detected if empty)
# CONVERT_OUTPUT: output directory for data.pt (auto-derived if empty)
TASK_NAME       ?= sitting_standing
ADAMIMIC_DATASET ?= ../../AdaMimic/legged_gym/resources/dataset/g1_dof27_data
CONVERT_INPUT   ?=
CONVERT_OUTPUT  ?= $(ADAMIMIC_DATASET)/$(TASK_NAME)/output

# --------------- Derived paths ---------------
# Strip path and extension from VIDEO_PATH, e.g. assets/sitting_standing.mp4 -> sitting_standing
VID_STEM        := $(basename $(notdir $(VIDEO_PATH)))
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

# --------------- jaxls solve env (Step 2 optimization & Step 4 retargeting) ---------------
# The jaxls FK-jacobian kernels crash XLA's compiler (both the CPU LLVM backend
# and the WSL2 CUDA plugin) at full optimization. Force CPU and lower the XLA
# optimization level so the solves compile cleanly (~90s for 100 timesteps).
# Only affects JAX; torch GPU work in these stages is unaffected.
JAXLS_ENV := JAX_PLATFORMS=cpu XLA_FLAGS="--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"

# --------------- Targets ---------------

.PHONY: pipeline all _init_dirs preprocess reconstruct optimize postprocess retarget visualize extract-frames convert-to-adamimic

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
	bash preprocess_human.sh $(VIDEO_PATH) $(VIS) && \
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
	$(JAXLS_ENV) python stage2_optimization/megahunter_optimization.py \
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
	$(JAXLS_ENV) python stage4_retargeting/robot_motion_retargeting.py \
	    --src-dir "$$POSTPROC_DIR" \
	    --contact-dir $(CONTACT_DIR) && \
	echo "Pipeline complete! Results in $$POSTPROC_DIR"

# Kept for backward compat; prefer `make pipeline`
all: _init_dirs preprocess reconstruct optimize postprocess retarget
	@echo "Pipeline complete! Results in $(POSTPROC_DIR)"

# ---------- Step 0: Preprocessing (vm1rs) ----------
preprocess: _init_dirs
	$(CONDA_EVAL) && conda activate vm1rs && \
	$(PROXY_ENV) bash preprocess_human.sh $(VIDEO_PATH) $(VIS)

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
	$(PROXY_ENV) $(JAXLS_ENV) python stage4_retargeting/robot_motion_retargeting.py \
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

# ---------- Convert to AdaMimic (vm1rs) ----------
# Converts retarget_poses_g1.h5 to AdaMimic data.pt for stage1 training.
# Usage:
#   make convert-to-adamimic                                          # auto-detect input
#   make convert-to-adamimic TASK_NAME=my_task                        # specify task name
#   make convert-to-adamimic CONVERT_INPUT=/path/to/retarget.h5       # specify input manually
#   make convert-to-adamimic CONVERT_SMOOTH=1                         # enable smoothing
CONVERT_SMOOTH ?=
# SOURCE_FPS: actual fps of the motion data.
# For a 15fps video with STRIDE=2, real fps = 15/2 = 7.5
# Override if your video has a different framerate.
SOURCE_FPS      ?=
convert-to-adamimic:
	@$(CONDA_EVAL) && conda activate vm1rs && \
	if [ -n "$(CONVERT_INPUT)" ]; then \
	    H5_PATH="$(CONVERT_INPUT)"; \
	else \
	    H5_PATH=$$(for d in $(BASE_DIR)/output_calib_mesh/*/; do [ -f "$$d/retarget_poses_g1.h5" ] && echo "$$d/retarget_poses_g1.h5" && break; done); \
	fi && \
	if [ -z "$$H5_PATH" ] || [ ! -f "$$H5_PATH" ]; then \
	    echo "Error: retarget_poses_g1.h5 not found. Run 'make retarget' first or set CONVERT_INPUT."; \
	    exit 1; \
	fi && \
	echo "Input:  $$H5_PATH" && \
	echo "Output: $(CONVERT_OUTPUT)/data.pt" && \
	echo "Task:   $(TASK_NAME)" && \
	mkdir -p $(CONVERT_OUTPUT) && \
	SMOOTH_FLAG=$$( [ "$(CONVERT_SMOOTH)" = "1" ] && echo "--smooth" || echo "" ) && \
	SOURCE_FPS_FLAG=$$( [ -n "$(SOURCE_FPS)" ] && echo "--source_fps $(SOURCE_FPS)" || echo "" ) && \
	python tools/convert_to_adamimic.py \
	    --input "$$H5_PATH" \
	    --output_dir $(CONVERT_OUTPUT) \
	    --task_name $(TASK_NAME) \
	    --target_fps 30 \
	    $$SMOOTH_FLAG \
	    $$SOURCE_FPS_FLAG && \
	JOINT_ID_SRC="$(ADAMIMIC_DATASET)/joint_id.txt" && \
	if [ ! -f "$(CONVERT_OUTPUT)/joint_id.txt" ] && [ -f "$$JOINT_ID_SRC" ]; then \
	    cp "$$JOINT_ID_SRC" "$(CONVERT_OUTPUT)/joint_id.txt" && \
	    echo "Copied joint_id.txt to $(CONVERT_OUTPUT)/"; \
	fi && \
	echo "Done. Run AdaMimic stage1 with: make -C ../../AdaMimic stage1 TASK=$(TASK_NAME)"

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
	$(JAXLS_ENV) python stage2_optimization/megahunter_optimization.py \
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
