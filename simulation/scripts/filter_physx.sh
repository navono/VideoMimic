#!/usr/bin/env bash
# Wrapper to run Isaac Gym training with PhysX noise filtered from console output.
# PhysX on RTX 4090 / Ada Lovelace can spam harmless C++ errors
# (internal CUDA errors, fetchResults called illegally, etc.) that make
# the console output unreadable. This script filters those lines while
# preserving all other console output and the wrapped command's exit code.
#
# Usage: ./filter_physx.sh <command> [args...]
#   e.g. ./filter_physx.sh torchrun --nproc-per-node 2 train.py ...

LOG_FILE="${FILTER_PHYSX_LOG:-${PWD}/logs/train-console-filtered.log}"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

filter_physx_noise() {
	sed '/^[[:space:]\r]*$/d
/internal error : PhysX/d
/PhysX Internal CUDA error/d
/internal error : GPU/d
/Could not find registered CUDA/d
/fetchResults() called illegally/d
/Not connected to PVD/d
/GPU Pipeline: enabled/d
/GPU MemCopy/d
/GPU radixSort/d
/GPU initialize ranks/d
/mergeChangedAABBMgrHandles/d
/PxgCudaBroadPhase/d
/PxgNarrowphase/d
/Physics Engine: PhysX/d
/Physics Device:/d
/invalid operation : PxScene/d'
}

set -o pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
printf 'Filtered console log: %s\n' "$LOG_FILE"
"$@" 2>&1 | filter_physx_noise | tee -a "$LOG_FILE"
exit "${PIPESTATUS[0]}"
