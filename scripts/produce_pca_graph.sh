#!/bin/bash


# Only one task at a time, pick between "Emotion" "Age" "Gender". Use the appropriate dataset:
# For age use UTK or FairFace, for emotion use RAF-DB. I would avoid Vgg, as it's size may cause problem when computing PCA components
TASK="Emotion"

# "trained", load the checkpoints, "untrained" do not load the checkpoint, useful if you want to visualize the raw features.
WEIGHTS="trained"

# Possible datasets = "RAF-DB", "FairFace", "UTK", "CelebA_HQ". "Vgg" is possible but not raccomended
DATASET="RAF-DB"

# type of model to load. Only peft model are supported for visualizations.
PEFT_TYPE="mtlora" # possible choices "lora" or "mtlora"
CHECKPOINT_PATH="./checkpoints/mtl_mtlora_ckpt" # make sure it match PEFT_TYPE


python -m src.pca \
    --task "$TASK" \
    --weights "$WEIGHTS" \
    --dataset "$DATASET" \
    --peft_type "$PEFT_TYPE" \
    --checkpoint "$CHECKPOINT_PATH"