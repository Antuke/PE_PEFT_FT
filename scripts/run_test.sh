#!/bin/bash

echo "Starting testing run..."

# --- Run & Path Settings ---
RUN_NAME="Test_Run_From_Config"
OUTPUT_DIR="./testing_outputs"
SCRIPT="-m src.test"

# IMPORTANT: Set the path to the trained model checkpoint and the configuration file
# Example:
# CKPT_PATH="./training_outputs/run_example/ckpt/model_1.pt"
# The checkpoint, in the case of using peft, is the folder that contains the adapter_model.pt file
# CONFIG_PATH="./training_outputs/run_example/config.json"
CKPT_PATH="."
CONFIG_PATH="."
LABELS_ROOT="./labels"
IMAGE_BASE_ROOT="."

# --- Dataset Settings ---
DATASET_NAMES="RAF-DB" # You can list multiple datasets, e.g., "UTK Vgg FairFace RAF-DB CelebA_HQ"

# --- Hyperparameter Settings ---
BATCH_SIZE=16

python ${SCRIPT} \
    --name "${RUN_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --setup_from_json "${CONFIG_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --output_dir "${OUTPUT_DIR}" \
    --labels_root "${LABELS_ROOT}" \
    --image_base_root "${IMAGE_BASE_ROOT}" \
    --dataset_names ${DATASET_NAMES}

echo "Testing run finished."
