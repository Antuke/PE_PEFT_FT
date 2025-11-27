#!/bin/bash

echo "Starting demo run..."

# --- Run & Path Settings ---
SCRIPT="-m src.demo"

# IMPORTANT: Set the path to the directory containing the trained model adapters
CKPT_PATH="./checkpoints/mtl_dora_ckpt"   # or you can set "./checkpoints/mtl_mtlora_ckpt" 
INPUT_DIR='./demo_images/'

# --- Task & Dataset Settings ---
# List the tasks to evaluate, separated by spaces.
TASKS="Gender Age Emotion"

# --- Model & PEFT Settings ---
# These should match the settings used during training for the loaded checkpoint.
PEFT_METHOD="--use_peft --use_deep_head" # add "--use_mtlora" if using mtlora_ckpt





python ${SCRIPT} \
    --ckpt_path "${CKPT_PATH}" \
    ${PEFT_METHOD} \
    --input_dir ${INPUT_DIR} \
    --tasks ${TASKS} \

echo "Demo run finished."