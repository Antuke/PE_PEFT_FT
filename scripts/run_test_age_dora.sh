#!/bin/bash

echo "Starting testing run..."

# --- Run & Path Settings ---
RUN_NAME="DoRA_Age_Test"
OUTPUT_DIR="./testing_outputs"
SCRIPT="-m src.test"
CKPT_PATH="./checkpoints/age_dora_ckpt"


LABELS_ROOT="./labels"
IMAGE_BASE_ROOT="."

# --- Task & Dataset Settings ---
# List the tasks to evaluate, separated by spaces.
TASKS="Age"
DATASET_NAMES="FairFace" # You can list multiple datasets, e.g., "RAF-DB CelebA"

# --- Model & PEFT Settings ---
# These should match the settings used during training for the loaded checkpoint.
PEFT_METHOD="--use_peft --use_deep_head"
RANK=64

# --- Hyperparameter Settings ---
BATCH_SIZE=32


python ${SCRIPT} \
    --name "${RUN_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --tasks ${TASKS} \
    ${PEFT_METHOD} \
    --batch_size ${BATCH_SIZE} \
    --output_dir "${OUTPUT_DIR}" \
    --labels_root "${LABELS_ROOT}" \
    --image_base_root "${IMAGE_BASE_ROOT}" \
    --repo_path "${PE_PATH}" \
    --dataset_names ${DATASET_NAMES}

echo "Testing run finished."