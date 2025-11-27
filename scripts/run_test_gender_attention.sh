#!/bin/bash

echo "Starting testing run..."

# --- Run & Path Settings ---
RUN_NAME="Attention_Gender_Test"
OUTPUT_DIR="./testing_outputs"
SCRIPT="-m src.test"
CKPT_PATH="./checkpoints/legacy_ckpt/gender_attention.pt"


LABELS_ROOT="./labels"
IMAGE_BASE_ROOT="."

# --- Task & Dataset Settings ---
# List the tasks to evaluate, separated by spaces.
TASKS="Gender"
DATASET_NAMES="FairFace" # You can list multiple datasets, e.g., "UTK Vgg CelebA_HQ"

# --- Model & PEFT Settings ---
# These should match the settings used during training for the loaded checkpoint.
PEFT_METHOD="--use_deep_head"

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