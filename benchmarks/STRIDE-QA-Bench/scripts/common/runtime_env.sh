#!/bin/bash
# Common runtime helpers for cluster + local
# Usage: source "$(git rev-parse --show-toplevel)/scripts/common/runtime_env.sh"

set -euo pipefail

# -----------------------------
# Logging
# -----------------------------
log_info()  { echo "[INFO]  $*"; }
log_warn()  { echo "[WARN]  $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

# -----------------------------
# Configurable defaults (can be overridden by env)
# -----------------------------
: "${CUDA_MODULE_VERSION:=12.4}"          # e.g., export CUDA_MODULE_VERSION=12.1
: "${USE_MODULES_DEFAULT:=auto}"          # auto|true|false
: "${ENABLE_DETERMINISM:=1}"              # 1 to export CUBLAS_WORKSPACE_CONFIG
: "${MASTER_PORT_BASE:=29500}"            # base port for torchrun
: "${MASTER_PORT_MAX:=29999}"

# -----------------------------
# Module-load helper (idempotent)
#   - Loads first "cuda/${CUDA_MODULE_VERSION}*" found by `module avail`
#   - Respects USE_MODULES_DEFAULT: auto|true|false
# -----------------------------
maybe_load_cuda_module() {
  case "${USE_MODULES_DEFAULT}" in
    false) log_info "'module' usage disabled (USE_MODULES_DEFAULT=false)"; return 0 ;;
    auto)
      if ! command -v module >/dev/null 2>&1; then
        log_info "'module' command not found. Skipping module load."
        return 0
      fi
      ;;
    true)
      if ! command -v module >/dev/null 2>&1; then
        log_warn "'module' forced but not found. Skipping."
        return 0
      fi
      ;;
    *)
      log_warn "Unknown USE_MODULES_DEFAULT=${USE_MODULES_DEFAULT}. Treat as 'auto'."
      if ! command -v module >/dev/null 2>&1; then
        log_info "'module' command not found. Skipping module load."
        return 0
      fi
      ;;
  esac

  log_info "Searching for CUDA ${CUDA_MODULE_VERSION} via 'module avail'..."
  # pick first match like cuda/12.4, cuda/12.4.1, prefix/*/cuda/12.4, etc.
  local m
  m="$(module avail 2>&1 | grep -oE '[^[:space:]]*cuda/'"${CUDA_MODULE_VERSION}"'([[:alnum:]._-]*)?' | head -n 1 || true)"
  if [[ -n "${m:-}" ]]; then
    log_info "Loading module ${m}"
    if module load "${m}"; then
      log_info "Loaded ${m}"
    else
      log_warn "Failed to load ${m}. Continuing without module load."
    fi
  else
    log_warn "No CUDA module matching 'cuda/${CUDA_MODULE_VERSION}' found. Continuing without module load."
  fi
}

# -----------------------------
# CUDA_HOME check (non-fatal)
# -----------------------------
check_cuda_home() {
  if [[ -z "${CUDA_HOME:-}" ]]; then
    log_warn "CUDA_HOME is not set. Assuming CUDA is available via PATH."
  else
    log_info "CUDA_HOME=${CUDA_HOME}"
    # if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    #   "${CUDA_HOME}/bin/nvcc" --version || true
    # else
    #   log_warn "nvcc not found under ${CUDA_HOME}/bin"
    # fi
  fi
}

# -----------------------------
# Determinism toggles (optional)
# -----------------------------
enable_determinism_if_requested() {
  if [[ "${ENABLE_DETERMINISM}" == "1" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    log_info "Determinism: CUBLAS_WORKSPACE_CONFIG set to :4096:8"
  fi
}

# -----------------------------
# Infer processes per node
#   Priority: SLURM_GPUS_ON_NODE -> nvidia-smi count -> 1
# -----------------------------
infer_nproc_per_node() {
  local nproc
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    nproc="${SLURM_GPUS_ON_NODE}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nproc="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${nproc}" =~ ^[0-9]+$ ]] || nproc=1
  else
    nproc=1
  fi
  echo "${nproc}"
}

# -----------------------------
# Pick a free-ish master port (deterministic-ish)
# -----------------------------
pick_master_port() {
  local salt="${SLURM_JOB_ID:-$RANDOM}"
  local span=$(( MASTER_PORT_MAX - MASTER_PORT_BASE ))
  local port=$(( MASTER_PORT_BASE + (salt % (span > 0 ? span : 1)) ))
  echo "${port}"
}

# -----------------------------
# Orchestrate environment setup
# -----------------------------
setup_runtime_env() {
  maybe_load_cuda_module
  check_cuda_home
  enable_determinism_if_requested

  # Export convenience vars if not present
  if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    export NPROC_PER_NODE="$(infer_nproc_per_node)"
    log_info "NPROC_PER_NODE=${NPROC_PER_NODE}"
  fi
  if [[ -z "${MASTER_PORT:-}" ]]; then
    export MASTER_PORT="$(pick_master_port)"
    log_info "MASTER_PORT=${MASTER_PORT}"
  fi
}