# Real2Sim Pipeline Makefile
# Usage:
#   make extract-frames VIDEO=video.mp4 NAME=my_video
#   make pipeline NAME=my_video
#   make pipeline NAME=my_video START=50 END=200
#   make setup-vm1rs
#   make verify-vm1rs
#
# Directory structure per video:
#   demo_data/<NAME>/
#   ├── cam01/           # extracted frames
#   ├── masks/cam01/     # SAM2 output
#   ├── poses2d/cam01/   # ViTPose output
#   ├── meshes3d/cam01/  # VIMO output
#   ├── contacts/cam01/  # BSTRO output
#   ├── megasam/         # MegaSam output
#   ├── megahunter/      # MegaHunter output
#   └── calib_mesh/      # postprocessing + retargeting output

SHELL := /bin/bash

# --- Configurable variables ---
NAME        ?= sitting_standing
START       ?= 0
END         ?= auto
STRIDE      ?= 1
ROBOT       ?= g1
HEIGHT      ?= -1
VIS         ?= 0
GENDER      ?= male
LOCALHUB    ?= 0

# --- Derived paths (all under demo_data/<NAME>/) ---
DATA_DIR    := demo_data/$(NAME)
CAM_DIR     := $(DATA_DIR)/cam01
MASKS_DIR   := $(DATA_DIR)/masks/cam01
POSE2D_DIR  := $(DATA_DIR)/poses2d/cam01
SMPL_DIR    := $(DATA_DIR)/meshes3d/cam01
CONTACT_DIR := $(DATA_DIR)/contacts/cam01
MEGASAM_DIR := $(DATA_DIR)/megasam
MH_OUT_DIR  := $(DATA_DIR)/megahunter
CALIB_DIR   := $(DATA_DIR)/calib_mesh

# --- Auto-detect frame count from cam01/ ---
# If END=auto, count jpg files in CAM_DIR; otherwise use the given value.
# Extracted frames are 1-indexed (00001.jpg, 00002.jpg, ...),
# so the 0-indexed end frame = frame_count - 1.
AUTO_END    := $(shell ls $(CAM_DIR)/*.jpg 2>/dev/null | wc -l)
ACTUAL_END  := $(if $(filter auto,$(END)),$(shell expr $(AUTO_END) - 1),$(END))

MEGASAM_H5  := $(MEGASAM_DIR)/megasam_reconstruction_results_$(NAME)_cam01_frame_$(START)_$(ACTUAL_END)_subsample_$(STRIDE).h5
MH_H5       := $(MH_OUT_DIR)/megahunter_megasam_reconstruction_results_$(NAME)_cam01_frame_$(START)_$(ACTUAL_END)_subsample_$(STRIDE).h5

# --- Conda helpers ---
CONDA_ACTIVATE_VM1RS     = eval "$$(conda shell.bash hook)" && conda activate vm1rs
CONDA_ACTIVATE_VM1RECON  = eval "$$(conda shell.bash hook)" && conda activate vm1recon

# --- CUDA paths (adjust if needed) ---
CUDA12 ?= /usr/local/cuda-12.4
CUDA11 ?= /usr/local/cuda-11.8

# ============================================================
# Environment Setup
# ============================================================

.PHONY: setup-vm1rs setup-vm1recon setup patch-vitpose

setup: setup-vm1rs setup-vm1recon patch-vitpose

## Setup vm1rs environment (Python 3.12, CUDA 12.4)
setup-vm1rs:
	@echo "=== Setting up vm1rs environment ==="
	conda create -n vm1rs python=3.12 -y
	$(CONDA_ACTIVATE_VM1RS) && \
	pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
	pip install viser tyro supervision transformers warp-lang timm einops scikit-learn boto3 requests pyliblzfse h5py yacs && \
	cd /tmp && git clone https://github.com/hongsukchoi/chumpy.git && cd chumpy && pip install --no-build-isolation -e . && cd /tmp && rm -rf chumpy && \
	pip install git+https://github.com/hongsukchoi/smplx && \
	mkdir -p third_party && cd third_party && \
	git clone https://github.com/hongsukchoi/Grounded-SAM-2.git && cd Grounded-SAM-2 && \
	CUDA_HOME=$(CUDA12) pip install -e . && \
	pip install --no-build-isolation -e grounding_dino && \
	pip install transformers && cd ../.. && \
	pip install --no-build-isolation mmcv==1.7.2 && \
	cd third_party && git clone https://github.com/ViTAE-Transformer/ViTPose.git && cd ViTPose && pip install -v -e . && cd ../.. && \
	pip install git+https://github.com/hongsukchoi/VIMO.git && \
	cd third_party && git clone --recursive https://github.com/hongsukchoi/bstro.git && cd bstro && pip install -e . && cd ../.. && \
	pip install "jax[cuda12]>=0.5,<0.6" && \
	pip install "git+https://github.com/brentyi/jaxls.git" && \
	cd third_party && git clone https://github.com/chungmin99/pyroki.git && cd pyroki && \
	git checkout 70b30a56b1e1ea83fb4c2cac8fe2c63a0624b9ce && pip install -e . && cd ../.. && \
	cd third_party && git clone https://github.com/nerfstudio-project/viser && cd viser && pip install -e . && cd ../..
	@echo "=== vm1rs setup complete ==="

## Setup vm1recon environment (Python 3.10, CUDA 11.8)
setup-vm1recon:
	@echo "=== Setting up vm1recon environment ==="
	cd third_party/megasam-package && conda env create -f environment.yml
	$(CONDA_ACTIVATE_VM1RECON) && \
	pip install viser tyro supervision transformers warp-lang timm einops scikit-learn boto3 requests pyliblzfse h5py yacs && \
	cd /tmp && git clone https://github.com/hongsukchoi/chumpy.git && cd chumpy && pip install --no-build-isolation -e . && cd /tmp && rm -rf chumpy && \
	pip install git+https://github.com/hongsukchoi/smplx && \
	pip install "numpy<2" && \
	pip install setuptools==69.5.1 && \
	CUDA_HOME=$(CUDA11) CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 pip install --no-build-isolation torch-scatter==2.1.2 && \
	cd third_party/megasam-package && \
	wget -q https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2 && \
	conda install -n vm1recon xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2 -y && \
	rm xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2 && cd ../.. && \
	cd third_party/megasam-package/base && \
	CUDA_HOME=$(CUDA11) CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 python setup.py install && cd ../../.. && \
	conda install -y -c pyg -c nvidia -c conda-forge pytorch-lightning=1.9.4 tensorboard pybind11 pyg rich pandas omegaconf && \
	pip install randomname pykdtree plyfile flatten-dict pyntcloud && \
	pip install python-pycg==0.5.2 --no-deps && pip install open3d && \
	cd /tmp && rm -rf nksr && git clone https://github.com/nv-tlabs/nksr.git && cd nksr/package && \
	CUDA_HOME=$(CUDA11) CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 pip install --no-build-isolation -e . && cd /tmp && rm -rf nksr && \
	pip install trimesh tyro h5py rtree && \
	git clone https://github.com/hongsukchoi/GeoCalib.git third_party/GeoCalib && cd third_party/GeoCalib && pip install -e . && cd ../.. && \
	pip install "wandb>=0.19" && \
	pip install "numpy<2"
	@echo "=== vm1recon setup complete ==="

## Patch ViTPose for numpy 2.x compatibility
patch-vitpose:
	@python -c "\
import pathlib; \
f = pathlib.Path('third_party/ViTPose/mmpose/models/heads/topdown_heatmap_base_head.py'); \
txt = f.read_text(); \
old = \"score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)\"; \
new = \"score[i] = float(np.array(img_metas[i]['bbox_score']).reshape(-1)[0])\"; \
txt = txt.replace(old, new); \
f.write_text(txt); \
print('ViTPose numpy 2.x patch applied.')"

# ============================================================
# Network Pre-processing (proxy environments)
# ============================================================

.PHONY: cache-dinov2 cache-geocalib network-setup

network-setup: cache-dinov2 cache-geocalib

## Cache dinov2 for --localhub (proxy/SSL workaround)
cache-dinov2:
	@echo "=== Caching dinov2 for --localhub ==="
	mkdir -p ~/.cache/torch/hub/torchhub
	cd ~/.cache/torch/hub/torchhub && \
	if [ ! -d facebookresearch_dinov2_main ]; then \
	    git clone https://github.com/facebookresearch/dinov2.git facebookresearch_dinov2_main; \
	fi
	mkdir -p torchhub
	ln -sf ~/.cache/torch/hub/torchhub/facebookresearch_dinov2_main torchhub/
	@echo "=== dinov2 cached ==="

## Cache GeoCalib model weights (proxy/SSL workaround)
cache-geocalib:
	@echo "=== Caching GeoCalib model weights ==="
	mkdir -p ~/.cache/torch/hub/geocalib
	@if [ ! -f ~/.cache/torch/hub/geocalib/pinhole.tar ]; then \
	    curl -L -o ~/.cache/torch/hub/geocalib/pinhole.tar \
	        https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar; \
	fi
	@if [ ! -f ~/.cache/torch/hub/geocalib/distorted.tar ]; then \
	    curl -L -o ~/.cache/torch/hub/geocalib/distorted.tar \
	        https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-distorted.tar; \
	fi
	@echo "=== GeoCalib weights cached ==="

# ============================================================
# Verification
# ============================================================

.PHONY: verify-vm1rs verify-vm1recon verify

verify: verify-vm1rs verify-vm1recon

## Verify vm1rs environment
verify-vm1rs:
	@echo "=== Verifying vm1rs ==="
	@$(CONDA_ACTIVATE_VM1RS) && python -c "\
import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}'); \
import mmcv; print(f'mmcv: {mmcv.__version__}'); \
from mmpose.apis import inference_top_down_pose_model, init_pose_model; \
import vimo; print('vimo: OK'); \
import metro; print('bstro(metro): OK'); \
import jax; print(f'jax: {jax.__version__}'); \
import jaxls; print('jaxls: OK'); \
import pyroki; print('pyroki: OK'); \
import sam2; print('sam2: OK'); \
import grounding_dino; print('grounding_dino: OK'); \
print('vm1rs: ALL OK')"

## Verify vm1recon environment
verify-vm1recon:
	@echo "=== Verifying vm1recon ==="
	@$(CONDA_ACTIVATE_VM1RECON) && python -c "\
import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}'); \
import xformers; print(f'xformers: {xformers.__version__}'); \
import torch_scatter; print(f'torch_scatter: {torch_scatter.__version__}'); \
import lietorch; print('lietorch (DROID-SLAM): OK'); \
import nksr; print('nksr: OK'); \
import geocalib; print('geocalib: OK'); \
import open3d; print('open3d: OK'); \
import numpy; print(f'numpy: {numpy.__version__}'); \
print('vm1recon: ALL OK')"

# ============================================================
# Pipeline Stages
# ============================================================

.PHONY: stage0 stage0-sam2 stage0-vitpose stage0-vimo stage0-bstro

## Stage 0: Full preprocessing (SAM2 + ViTPose + VIMO + BSTRO)
stage0:
	@echo "=== Stage 0: Preprocessing (vm1rs) ==="
	@$(CONDA_ACTIVATE_VM1RS) && \
	if [ "$(VIS)" = "1" ]; then VIS_FLAG="--vis"; else VIS_FLAG=""; fi && \
	echo "Running SAM2..." && \
	python stage0_preprocessing/sam2_segmentation.py \
	    --video-dir "$(CAM_DIR)" \
	    --output-dir "$(MASKS_DIR)" \
	    $${VIS_FLAG} && \
	echo "Running ViTPose..." && \
	python stage0_preprocessing/vitpose_2d_poses.py \
	    --video-dir "$(CAM_DIR)" \
	    --bbox-dir "$(MASKS_DIR)/json_data" \
	    --output-dir "$(POSE2D_DIR)" \
	    $${VIS_FLAG} && \
	echo "Running VIMO..." && \
	python stage0_preprocessing/vimo_3d_mesh.py \
	    --img-dir "$(CAM_DIR)" \
	    --mask-dir "$(MASKS_DIR)" \
	    --out-dir "$(SMPL_DIR)" && \
	echo "Running BSTRO..." && \
	python stage0_preprocessing/bstro_contact_detection.py \
	    --video-dir "$(CAM_DIR)" \
	    --bbox-dir "$(MASKS_DIR)/json_data" \
	    --output-dir "$(CONTACT_DIR)" \
	    --feet-contact-ratio-thr 0.2 \
	    --contact-thr 0.95 && \
	echo "=== Stage 0 complete ==="

stage0-sam2:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python stage0_preprocessing/sam2_segmentation.py \
	    --video-dir "$(CAM_DIR)" --output-dir "$(MASKS_DIR)" --vis

stage0-vitpose:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python stage0_preprocessing/vitpose_2d_poses.py \
	    --video-dir "$(CAM_DIR)" --bbox-dir "$(MASKS_DIR)/json_data" --output-dir "$(POSE2D_DIR)" --vis

stage0-vimo:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python stage0_preprocessing/vimo_3d_mesh.py \
	    --img-dir "$(CAM_DIR)" --mask-dir "$(MASKS_DIR)" --out-dir "$(SMPL_DIR)"

stage0-bstro:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python stage0_preprocessing/bstro_contact_detection.py \
	    --video-dir "$(CAM_DIR)" --bbox-dir "$(MASKS_DIR)/json_data" --output-dir "$(CONTACT_DIR)" \
	    --feet-contact-ratio-thr 0.2 --contact-thr 0.95

# --- Stage 1: MegaSam reconstruction (vm1recon) ---

.PHONY: stage1

stage1:
	@echo "=== Stage 1: MegaSam reconstruction (vm1recon) ==="
	@echo "Frames: $(START) - $(ACTUAL_END) (auto-detected from $(CAM_DIR))"
	@$(CONDA_ACTIVATE_VM1RECON) && \
	if [ "$(LOCALHUB)" = "1" ]; then LOCALHUB_FLAG="--localhub"; else LOCALHUB_FLAG=""; fi && \
	python stage1_reconstruction/megasam_reconstruction.py \
	    --video-dir "$(CAM_DIR)" \
	    --out-dir "$(MEGASAM_DIR)" \
	    --start-frame $(START) \
	    --end-frame $(ACTUAL_END) \
	    --stride $(STRIDE) \
	    --gsam2 \
	    $${LOCALHUB_FLAG} && \
	echo "=== Stage 1 complete ==="

# --- Stage 2: MegaHunter optimization (vm1rs) ---

.PHONY: stage2

stage2:
	@echo "=== Stage 2: MegaHunter optimization (vm1rs) ==="
	@$(CONDA_ACTIVATE_VM1RS) && \
	if [ "$(HEIGHT)" != "-1" ] && [ "$(HEIGHT)" != "0" ]; then \
	    echo "Optimizing SMPL shape for height=$(HEIGHT)..." && \
	    python stage2_optimization/optimize_smpl_shape_for_height.py \
	        --height $(HEIGHT) --output-dir "$(SMPL_DIR)"; \
	fi && \
	MH_FLAGS="" && \
	if [ "$(HEIGHT)" = "0" ] && [ "$(ROBOT)" = "g1" ]; then \
	    MH_FLAGS="--use-g1-shape"; \
	    echo "Using g1 shape"; \
	fi && \
	python stage2_optimization/megahunter_optimization.py \
	    --world-env-path "$(MEGASAM_H5)" \
	    --bbox-dir "$(MASKS_DIR)/json_data" \
	    --pose2d-dir "$(POSE2D_DIR)" \
	    --smpl-dir "$(SMPL_DIR)" \
	    --out-dir "$(MH_OUT_DIR)" \
	    $${MH_FLAGS} && \
	echo "=== Stage 2 complete ==="

# --- Stage 3: Postprocessing - GeoCalib + NKSR (vm1recon) ---

.PHONY: stage3

stage3:
	@echo "=== Stage 3: Postprocessing (vm1recon) ==="
	@$(CONDA_ACTIVATE_VM1RECON) && \
	python stage3_postprocessing/postprocessing_pipeline.py \
	    --megahunter-path "$(MH_H5)" \
	    --out-dir "$(CALIB_DIR)" \
	    --gender $(GENDER) \
	    --is-megasam && \
	echo "=== Stage 3 complete ==="

# --- Stage 4: Robot motion retargeting (vm1rs) ---

.PHONY: stage4

stage4:
	@echo "=== Stage 4: Robot retargeting (vm1rs) ==="
	@$(CONDA_ACTIVATE_VM1RS) && \
	python stage4_retargeting/robot_motion_retargeting.py \
	    --src-dir "$(CALIB_DIR)" \
	    --contact-dir "$(CONTACT_DIR)" && \
	echo "=== Stage 4 complete ==="

# ============================================================
# Full Pipeline
# ============================================================

.PHONY: pipeline

## Run the complete pipeline (Stage 0 -> 4)
pipeline: stage0 stage1 stage2 stage3 stage4
	@echo ""
	@echo "============================================"
	@echo " Pipeline complete!"
	@echo " Output: $(DATA_DIR)/"
	@echo "============================================"

# ============================================================
# Visualization
# ============================================================

.PHONY: vis-env vis-megahunter vis-calib vis-mesh vis-retarget vis-results

## Visualize MegaSam environment reconstruction
vis-env:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python visualization/environment_only_visualization.py \
	    --world-env-path "$(MEGASAM_H5)" \
	    --world-scale-factor 1.5 --conf-thr 0.0

## Visualize MegaHunter optimization results
vis-megahunter:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python visualization/optimization_results_visualization.py \
	    --world-env-path "$(MH_H5)" --bg-pc-downsample-factor 4

## Visualize gravity calibration results
vis-calib:
	@$(CONDA_ACTIVATE_VM1RECON) && \
	python visualization/gravity_calibration_visualization.py \
	    --calib-out-dir "$(CALIB_DIR)"

## Visualize mesh generation results
vis-mesh:
	@$(CONDA_ACTIVATE_VM1RECON) && \
	python visualization/mesh_generation_visualization.py \
	    --mesh-path "$(CALIB_DIR)/background_mesh.obj" \
	    --points-path "$(CALIB_DIR)/background_more_filtered_colored_pointcloud.ply"

## Visualize retargeting results
vis-retarget:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python visualization/retargeting_visualization.py \
	    --postprocessed-dir "$(CALIB_DIR)" \
	    --robot-name $(ROBOT) --bg-pc-downsample-factor 4 --confidence-threshold 0.0

## Visualize complete results with ego-view
vis-results:
	@$(CONDA_ACTIVATE_VM1RS) && \
	python visualization/complete_results_egoview_visualization.py \
	    --postprocessed-dir "$(CALIB_DIR)" \
	    --robot-name $(ROBOT) --bg-pc-downsample-factor 4 --is-megasam --save-ego-view

# ============================================================
# Utilities
# ============================================================

.PHONY: extract-frames count-frames info help

## Show detected frame info for current NAME
info:
	@echo "NAME:       $(NAME)"
	@echo "DATA_DIR:   $(DATA_DIR)"
	@echo "CAM_DIR:    $(CAM_DIR)"
	@echo "Frame count: $(AUTO_END)"
	@echo "END (0-indexed): $(ACTUAL_END)"
	@echo "Output h5:  $(MEGASAM_H5)"

## Count frames in a video or extracted image directory
count-frames:
	@if [ -n "$(VIDEO)" ]; then \
	    echo -n "Video frame count: " && \
	    ffprobe -v error -count_frames -select_streams v:0 \
	        -show_entries stream=nb_read_frames -of csv=p=0 "$(VIDEO)"; \
	elif [ -d "$(CAM_DIR)" ]; then \
	    echo "Extracted frames in $(CAM_DIR): $(AUTO_END)"; \
	else \
	    echo "Usage: make count-frames VIDEO=path/to/video.mp4"; \
	    echo "       make count-frames NAME=my_video"; \
	fi

## Extract ALL frames from video (no need to specify END)
extract-frames:
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make extract-frames VIDEO=path/to/video.mp4 NAME=my_video"; exit 1; fi
	@TOTAL=$$(ffprobe -v error -count_frames -select_streams v:0 \
	    -show_entries stream=nb_read_frames -of csv=p=0 "$(VIDEO)" | tr -d '[:space:]') && \
	END_IDX=$$((TOTAL - 1)) && \
	echo "Video has $$TOTAL frames, extracting frames $(START)-$$END_IDX..." && \
	$(CONDA_ACTIVATE_VM1RS) && \
	python utilities/extract_frames_from_video.py \
	    --video-path "$(VIDEO)" \
	    --output-dir "$(DATA_DIR)/cam01" \
	    --start-frame $(START) --end-frame $$END_IDX

## Show help
help:
	@echo "Real2Sim Pipeline Makefile"
	@echo ""
	@echo "All data for a video is stored under: demo_data/<NAME>/"
	@echo ""
	@echo "  demo_data/<NAME>/"
	@echo "  ├── cam01/           # extracted frames"
	@echo "  ├── masks/cam01/     # SAM2 output"
	@echo "  ├── poses2d/cam01/   # ViTPose output"
	@echo "  ├── meshes3d/cam01/  # VIMO output"
	@echo "  ├── contacts/cam01/  # BSTRO output"
	@echo "  ├── megasam/         # MegaSam output"
	@echo "  ├── megahunter/      # MegaHunter output"
	@echo "  └── calib_mesh/      # postprocessing + retargeting output"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make setup-vm1rs          Setup vm1rs environment"
	@echo "  make setup-vm1recon       Setup vm1recon environment"
	@echo "  make setup                Setup both environments + patch ViTPose"
	@echo "  make patch-vitpose        Patch ViTPose for numpy 2.x"
	@echo "  make network-setup        Cache dinov2 + GeoCalib (proxy workaround)"
	@echo ""
	@echo "Verification:"
	@echo "  make verify-vm1rs         Verify vm1rs imports"
	@echo "  make verify-vm1recon      Verify vm1recon imports"
	@echo ""
	@echo "Pipeline (END defaults to auto-detect):"
	@echo "  make extract-frames VIDEO=x.mp4 NAME=my_video   Extract frames"
	@echo "  make pipeline NAME=my_video                     Run all stages"
	@echo "  make stage0 NAME=my_video                       Preprocessing"
	@echo "  make stage1 NAME=my_video                       MegaSam reconstruction"
	@echo "  make stage2 NAME=my_video                       MegaHunter optimization"
	@echo "  make stage3 NAME=my_video                       Postprocessing"
	@echo "  make stage4 NAME=my_video                       Robot retargeting"
	@echo ""
	@echo "  Optional overrides: START=0 END=auto STRIDE=1 ROBOT=g1 HEIGHT=-1"
	@echo "  HEIGHT=0 uses G1 body shape; HEIGHT=1.8 fits SMPL to 1.8m"
	@echo "  LOCALHUB=1 for proxy environments"
	@echo ""
	@echo "Visualization:"
	@echo "  make vis-env              Visualize MegaSam environment"
	@echo "  make vis-megahunter       Visualize MegaHunter results"
	@echo "  make vis-calib            Visualize gravity calibration"
	@echo "  make vis-mesh             Visualize mesh generation"
	@echo "  make vis-retarget         Visualize retargeting results"
	@echo "  make vis-results          Visualize complete results (ego-view)"
	@echo ""
	@echo "Utilities:"
	@echo "  make info NAME=my_video   Show detected frame info"
	@echo "  make count-frames VIDEO=x.mp4  Count frames in video"
	@echo "  make count-frames NAME=my_video  Count extracted frames"
	@echo ""
	@echo "Examples:"
	@echo "  make extract-frames VIDEO=video.mp4 NAME=sitting_standing"
	@echo "  make pipeline NAME=sitting_standing"
	@echo "  make pipeline NAME=sitting_standing STRIDE=2 HEIGHT=0"
	@echo "  make stage0 NAME=sitting_standing VIS=1"
