#!/bin/bash


echo "Starting training run..."


# Name for the output folder
# Where training log, checkpoints and confusion matrices get saved
RUN_NAME="DoRA_TRAIN"    

# --- Task & Dataset Settings ---

# List the tasks to train on, separated by spaces.
# Valid options: Age, Gender, Emotion
TASKS="Gender Emotion"
SCRIPT="-m src.train"
# List the dataset to train on, separated by spaces.
# Valid options: RAF-DB FairFace Lagenda Celeba_HQ
# (Be sure to have unzipped the chosen datasets in the datasets_with_standard_labels folder)
DATASET_NAMES="RAF-DB"


# Workers to instanciate for DataLoader (if possible, set it to a value more than 1, to considerably speed up)
NUM_WORKERS=2

# --- Model & PEFT Settings ---
# Vision encoder to use, choose from: PE-Core-G14-448, PE-Core-L14-336, PE-Core-B16-224, PE-Core-S16-384, PE-Core-T16-384
# Keep in mind of the model sizes, G14 is the largest, T16 is the smallest. All the training/testing that has reported in this work has been done with PE-Core-L14-336
BACKBONE_NAME="PE-Core-L14-336"


# Specify the model settings.
# --use_deep_head --> The classification head will have an addition hidden layer, with the same size as the input
# --use_peft --> Will add DoRA modules to the backbone
# --use_mtlora --> The backbone will have task specific LoRA modules in the last transformer block and in the attention pooling layer (ignored if the training is single task)
# --lp --> Only the classification head is trained
# --ap --> Only the classification head and the attention pooling layer is trained

# Can be empty, for example if you want to train by only unfreezing layers.
TRAINING_METHOD="--use_peft --use_deep_head"


# The last [int] layer, including the attention pooling and classification head will be trained.
# If using peft, only the peft modules at those layer will be trained. If you want to train all 
# layers (with or without peft) set this to an high number like 99, keeping in mind the VRAM requirements. Ignored if using lp or ap
UNFREEZE_LAYERS=99
# Still, some parameters will always not be updated, like the first embedding layers, and the first layer normalization.


# Hyperparameter Settings 
EPOCHS=20
BATCH_SIZE=16
LEARNING_RATE=1e-4

# Weight decay for the AdamW optimizers
WEIGHT_DECAY=0.01

# rank of the peft modules (ignored if not using peft)
RANK=64

# The learning rate of the B LoRA matrices will be multiplied by this value
LORA_PLUS_LAMBDA=6

# The learning rate of the parameters of the backbone (even the peft modules) will be multiplied by this value
BACKBONE_LR_RATIO=0.1

# Specify the training settings.
# --use_grad_checkpointing --> Use grad checkpointing reduces the VRAM requirements, at the cost of having a slower training.
# --use_running_means --> Use running means to balance task loss to have a balanced multi-task loss (ignored if the training is single task)
# --use_uw --> Use uncertainty weighting to balance task loss (ignored if the training is single task)
# --balanced_sampling --> Intra-class balanced sampling. Useful when training on Age and Emotion. Uses inverse frequency to score and sample more frequently less represented labels.
# --use_early_stopping --> Stop training early when validation loss or/and accuracy does not improve for 10 epoch
# --use_avg_val_to_early_stop --> Uses validation loss to evaluate early stop criteria
# --use_avg_acc_to_early_stop --> Uses accuracy to evaluate early stop criteria
# --compile_model --> Enables model compilation (Warning, it may causes warnings, this means that something has gone wrong and the model will not be compiled,
#                     if this happens, you may want to change backend or disable this flag, to change backend, go in train.py and find and modify model = torch.compile(model,backend=''))
TRAINING_FLAGS="--use_uw --use_early_stopping --use_avg_val_to_early_stop --balanced_sampling --use_grad_checkpointing"


# --- Path Settings ---
# Adjust these paths to match your project structure if needed (should not be necessary if following the default structure)
OUTPUT_DIR="./training_outputs"
LABELS_ROOT="./labels"
IMAGE_BASE_ROOT="."




python ${SCRIPT} \
    --name "${RUN_NAME}" \
    --tasks ${TASKS} \
    --backbone_name ${BACKBONE_NAME} \
    ${TRAINING_METHOD} \
    --rank ${RANK} \
    --unfreeze_layers ${UNFREEZE_LAYERS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    ${TRAINING_FLAGS} \
    --output_dir "${OUTPUT_DIR}" \
    --labels_root "${LABELS_ROOT}" \
    --image_base_root "${IMAGE_BASE_ROOT}" \
    --dataset_names ${DATASET_NAMES} \
    --num_workers ${NUM_WORKERS} \
    --backbone_lr_ratio ${BACKBONE_LR_RATIO}

    

