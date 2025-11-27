"""
Code for training. It parses a series of argument and instanciate the model, its wrapper, the dataset classes
and starts the training loop
"""

import matplotlib.pyplot as plt
import os
import argparse
import json
import csv
import random
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    LambdaLR,
    CosineAnnealingLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import seaborn as sns
from utils.commons import *
from utils.dataset import *
from utils.task_config import Task
from src.utils.config_parser import parse_config
from src.model import MTLModel
from src.running_means import RunningMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import torch._dynamo

torch._dynamo.config.suppress_errors = True


def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Task Learning Training Script")

    # Paths and environment
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default="/mnt/c/Users/antonio/Desktop/datasets",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--image_base_root",
        type=str,
        default=".",
        help="Base root for images",
    )
    parser.add_argument(
        "--labels_root",
        type=str,
        default="./labels",
        help="Root directory for labels",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./training_outputs",
        help="Directory to save outputs",
    )

    # Training hyperparameters
    parser.add_argument(
        "--name", type=str, default="training", help="Name of the training run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use for training (e.g., "cuda", "cpu")',
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--rank", type=int, default=64, help="Rank for LoRA")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--backbone_lr_ratio",
        type=float,
        default=0.1,
        help="Learning rate ratio for the backbone",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--amsgrad", action="store_true", help="Use amsgrad")
    # PEFT parameters
    parser.add_argument("--use_mtlora", action="store_true", help="Use MT-LoRA")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT")
    parser.add_argument(
        "--lora_plus_lambda", type=int, default=6, help="Lambda for LoRA+"
    )

    # Training type
    parser.add_argument("--lp", action="store_true", help="Linear probing")
    parser.add_argument("--ap", action="store_true", help="Attention probing")
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=99,
        help="Number of layers to unfreeze from the end",
    )

    # MTL loss balancing
    parser.add_argument(
        "--use_uw",
        action="store_true",
        help="Use Uncertainty Weighting for loss balancing",
    )
    parser.add_argument(
        "--use_running_means",
        action="store_true",
        help="Use running means for loss balancing",
    )
    parser.add_argument(
        "--ema_alpha", type=float, default=0.95, help="EMA alpha for running means"
    )

    # Model and scheduler
    parser.add_argument(
        "--backbone_name",
        type=str,
        choices=[
            "PE-Core-L14-336",
            "PE-Core-B16-224",
            "PE-Core-S16-384",
            "PE-Core-T16-384",
            "PE-Core-G14-448"
        ],
        default="PE-Core-L14-336",
        help="Name of the backbone model",
    )
    parser.add_argument(
        "--attention_specific_pool",
        action="store_true",
        help="Use task-specific attention pooling",
    )
    parser.add_argument(
        "--use_deep_head",
        action="store_true",
        help="Use a prediction head with one hidden layer",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau scheduler",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.1,
        help="Factor for ReduceLROnPlateau scheduler",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "warmup-cosine"],
        help="LR Scheduler to use, choises = cosine, plateau, warmup-cosine",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for linear warmup (used with warmup-cosine scheduler)",
    )

    # Early stopping
    parser.add_argument(
        "--use_early_stopping", action="store_true", help="Enable early stopping"
    )
    parser.add_argument(
        "--use_avg_val_to_early_stop",
        action="store_true",
        help="Use average validation loss for early stopping",
    )
    parser.add_argument(
        "--use_avg_acc_to_early_stop",
        action="store_true",
        help="Use average accuracy for early stopping",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )

    # Dataset
    parser.add_argument(
        "--dataset_names", nargs="+", default=["RAF-DB"], help="List of dataset names"
    )
    parser.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="Enable intra-task balanced class sampling",
    )

    parser.add_argument("--compile_model", action="store_true", help="Compile model")
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        help="Uses grad-checkpointing to save VRAM",
    )

    # Tasks
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Age", "Gender", "Emotion"],
        choices=["Age", "Gender", "Emotion"],
        help="Specify which tasks to train on. Example: --tasks Age Gender",
    )
    parser.add_argument("--resume_checkpoint_state", type=str, default=None)

    return parse_config(parser)


def save_training_state(save_path_dir, optimizer, scheduler, scaler, epoch):
    """Saves the optimizer, scheduler, scaler, and epoch states to a file."""
    os.makedirs(save_path_dir, exist_ok=True)
    try:
        state = {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }

        state_file_path = os.path.join(save_path_dir, "training_state.pth")
        torch.save(state, state_file_path)
        print(f"Saved training state to {state_file_path}")
    except Exception as e:
        print(f"Error saving training state: {e}")


def setup_tasks(args):
    """Configures and returns the list of tasks based on arguments."""
    ALL_POSSIBLE_TASKS = [
        Task(
            name="Age",
            class_labels=[
                "0-2",
                "3-9",
                "10-19",
                "20-29",
                "30-39",
                "40-49",
                "50-59",
                "60-69",
                "70+",
            ],
            criterion=torch.nn.CrossEntropyLoss,
            weight=1.0,
            use_weighted_loss=False,
        ),
        Task(
            name="Gender",
            class_labels=["Male", "Female"],
            criterion=torch.nn.CrossEntropyLoss,
            weight=1.0,
            use_weighted_loss=False,
        ),
        Task(
            name="Emotion",
            class_labels=[
                "Surprise",
                "Fear",
                "Disgust",
                "Happy",
                "Sad",
                "Angry",
                "Neutral",
            ],
            criterion=torch.nn.CrossEntropyLoss,
            weight=1.0,
            use_weighted_loss=False,
        ),
    ]

    # Filter the list of tasks based on the command-line argument
    TASKS = [task for task in ALL_POSSIBLE_TASKS if task.name in args.tasks]
    if not TASKS:
        raise ValueError(
            "No valid tasks selected. Please choose from 'Age', 'Gender', 'Emotion'."
        )
    return TASKS


def setup_directories(args, tasks):
    """Creates output directories and saves configuration."""
    run_name = f"run_{args.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = args.output_dir / run_name
    ckpt_dir = output_dir / "ckpt"
    cm_dir = output_dir / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    cm_dir.mkdir(exist_ok=True)
    for task in tasks:
        (cm_dir / task.name).mkdir(exist_ok=True)

    config = vars(args)
    config.update(
        {
            "tasks": [
                {
                    "name": task.name,
                    "class_labels": task.class_labels,
                    "criterion": task.criterion.__name__,
                    "weight": task.weight,
                    "use_weighted_loss": task.use_weighted_loss,
                }
                for task in tasks
            ]
        }
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, default=str)
    
    print(f"Outputs will be saved to: {output_dir}")
    print("Configuration saved to config.json")
    
    return output_dir, ckpt_dir, cm_dir


def setup_model(args, tasks, device):
    """Initializes the model, handles freezing/unfreezing, and compilation."""
    backbone, transform, _ = get_backbone(args.backbone_name, apply_migration=True)
    res_block_layers = len(backbone.transformer.resblocks)
    layer_cutoff_idx = res_block_layers - args.unfreeze_layers
    model = MTLModel(
        backbone,
        tasks=tasks,
        rank=args.rank,
        truncate_idx=(res_block_layers - 2),
        last_lora_layers=layer_cutoff_idx,
        use_mtl_lora=args.use_mtlora,
        use_lora=args.use_peft,
        use_deep_head=args.use_deep_head,
        use_batch_norm=(not args.lp),
        use_dora=True,
    ).to(device)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the module targeted for training
    if args.use_mtlora == True:
        for name, param in model.named_parameters():
            if "lora" in name or "prediction_layers" in name or "probe" in name:
                param.requires_grad = True
    elif args.use_peft == True:
        for name, param in model.named_parameters():
            if "lora" in name or "prediction_layers" in name:
                param.requires_grad = True
    elif args.lp == True:
        for name, param in model.named_parameters():
            if "prediction_layers" in name:
                param.requires_grad = True
    elif args.ap == True:
        for name, param in model.named_parameters():
            if "attn_pool" in name or "prediction_layers" in name:
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if "attn_pool" in name or "prediction_layers" in name:
                param.requires_grad = True
            elif "backbone.transformer.resblocks" in name:
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_cutoff_idx:
                    param.requires_grad = True
            else:
                print(f'skipped: {name}')

    if args.compile_model == True:
        print(
            "Compiling Model...\n[IMPORTANT] You may see warnings due to compilation. The first batch will take a while to process"
        )
        model = torch.compile(model)

    return model, transform


def setup_optimizer_and_scheduler(args, model, device):
    """Sets up the optimizer, scaler, and scheduler."""
    print("==================PARAM GROUPS==================")
    head_params = []
    lora_B_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "attn_pool" in n or "prediction_layers" in n:
                print(f"head = {n}")
                head_params.append(p)
            elif "lora_B" in n or "lora_tasks_B" in n or "lora_shared_B" in n:
                print(f"b = {n}")
                lora_B_params.append(p)
            else:
                print(f"a = {n}")
                backbone_params.append(p)
    print("====================================")
    if args.use_uw:
        model.log_vars.requires_grad = True
        head_params.append(model.log_vars)

    optimizer_grouped_parameters = [
        {"params": head_params, "lr": args.learning_rate},
        {
            "params": lora_B_params,
            "lr": args.learning_rate * args.backbone_lr_ratio * args.lora_plus_lambda,
        },
        {"params": backbone_params, "lr": args.learning_rate * args.backbone_lr_ratio},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    scaler = torch.amp.GradScaler(device)
    
    scheduler = None
    if args.scheduler_type == "plateau":
        mode = "min" if args.use_avg_val_to_early_stop else "max"
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=[
                args.learning_rate * args.scheduler_factor * 0.09,
                args.learning_rate
                * args.backbone_lr_ratio
                * args.lora_plus_lambda
                * args.scheduler_factor
                * 0.09,
                args.learning_rate
                * args.backbone_lr_ratio
                * args.scheduler_factor
                * 0.09,
            ], 
        )
    elif args.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    elif args.scheduler_type == "warmup-cosine":
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / float(max(1, args.warmup_epochs)),
        )  
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
        
    start_epoch = 0
    if args.resume_checkpoint_state:
        try:
            checkpoint = torch.load(args.resume_checkpoint_state, map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
        except Exception as e:
            print(f"Could not load state: {e}")
            
    if args.use_grad_checkpointing:
        model.enable_gradient_checkpointing()
        
    return optimizer, scaler, scheduler, start_epoch


def setup_dataloaders(args, tasks, transform, device):
    """Creates training and validation dataloaders."""
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)
            ),
            *transform.transforms,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )

    for arg, value in vars(args).items():
        print(f"{arg:<20}: {value}")
        
    TASK_NAMES = [task.name for task in tasks]
    NUM_TASKS = len(tasks)
    AGE_IDX, GENDER_IDX, EMOTION_IDX = 0, 1, 2

    if args.balanced_sampling:
        balance_task = (
            {EMOTION_IDX: 0.334} if "Emotion" in TASK_NAMES and NUM_TASKS > 1 else None
        )

        train_dataset = TaskBalanceDataset(
            dataset_names=args.dataset_names,
            transform=train_transforms,
            split="train",
            datasets_root=args.labels_root,
            image_base_root=args.image_base_root,
            verbose=True,
            balance_task=balance_task,
            augment_duplicate=None,
        )
        age_weights = train_dataset.get_class_weights(AGE_IDX, "default")
        gender_weights = train_dataset.get_class_weights(GENDER_IDX, "default")
        emotion_weights = train_dataset.get_class_weights(EMOTION_IDX, "default")
        class_weights = [age_weights, gender_weights, emotion_weights]
        train_sampler, _ = build_weighted_sampler(
            dataset=train_dataset,
            class_weights_per_task=class_weights,
            device=device,
            combine="mean",
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_dataset = MultiDataset(
            dataset_names=args.dataset_names,
            transform=train_transforms,
            split="train",
            datasets_root=args.labels_root,
            image_base_root=args.image_base_root,
            verbose=True,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    val_dataset = MultiDataset(
        dataset_names=args.dataset_names,
        transform=transform,
        split="val",
        datasets_root=args.labels_root,
        image_base_root=args.image_base_root,
        verbose=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_epoch(epoch, args, model, train_loader, device, optimizer, scaler, criterions, tasks, task_name_to_idx, running_means):
    """Handles the training loop for a single epoch."""
    model.train()
    running_train_loss = 0.0
    task_train_losses = {task.name: 0.0 for task in tasks}
    task_train_counts = {task.name: 0 for task in tasks}

    task_weights = {}
    if args.use_running_means and not args.use_uw:
        raw_weights = [
            1.0 / max(running_means.get_by_index(i) or 1.0, 1e-8)
            for i in range(len(tasks))
        ]
        avg_raw_weight = sum(raw_weights) / len(raw_weights)
        final_weights = [w / avg_raw_weight for w in raw_weights]
        task_weights = {task.name: final_weights[i] for i, task in enumerate(tasks)}
    elif not args.use_uw:
        task_weights = {task.name: task.weight for task in tasks}

    progress_bar_train = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"
    )
    for images, labels in progress_bar_train:
        images, labels = images.to(device), labels.to(device)
        gt_labels = {
            task.name: labels[:, task_name_to_idx[task.name]] for task in tasks
        }
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            outputs = model(images)
            total_loss = 0
            current_batch_task_losses = {}
            for idx, task in enumerate(tasks):
                valid_indices = gt_labels[task.name] != -100
                if torch.any(valid_indices):
                    task_loss = criterions[task.name](
                        outputs[task.name], gt_labels[task.name]
                    )
                    current_batch_task_losses[task.name] = task_loss
                    task_train_losses[task.name] += (
                        task_loss.item() * valid_indices.sum().item()
                    )
                    task_train_counts[task.name] += valid_indices.sum().item()
                    if args.use_running_means:
                        running_means.update_by_idx(task_loss.item(), idx)

            if args.use_uw:
                for task_name, task_loss in current_batch_task_losses.items():
                    task_idx = task_name_to_idx[task_name]
                    precision = torch.exp(-model.log_vars[task_idx])
                    total_loss += (
                        precision * task_loss + 0.5 * model.log_vars[task_idx]
                    )
            else:
                for task_name, task_loss in current_batch_task_losses.items():
                    total_loss += task_weights[task_name] * task_loss

        if isinstance(total_loss, torch.Tensor) and total_loss != 0:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += total_loss.item()
        progress_bar_train.set_postfix(
            loss=f"{running_train_loss / (progress_bar_train.n + 1):.4f}"
        )
        
    avg_train_loss = (
        running_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    )
    avg_task_train_losses = {
        name: (task_train_losses[name] / task_train_counts[name])
        if task_train_counts[name] > 0
        else 0.0
        for name in [t.name for t in tasks]
    }
    
    return avg_train_loss, avg_task_train_losses, task_weights


def validate_epoch(epoch, args, model, val_loader, device, criterions, tasks, task_name_to_idx, cm_dir):
    """Handles validation and confusion matrix generation."""
    model.eval()
    task_val_losses = {task.name: 0.0 for task in tasks}
    task_val_counts = {task.name: 0 for task in tasks}
    all_preds = {task.name: [] for task in tasks}
    all_labels = {task.name: [] for task in tasks}
    TASK_NAMES = [task.name for task in tasks]

    with torch.no_grad():
        progress_bar_val = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"
        )
        for images, labels in progress_bar_val:
            images, labels = images.to(device), labels.to(device)
            gt_labels = {
                task.name: labels[:, task_name_to_idx[task.name]] for task in tasks
            }
            
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=True
            ):
                outputs = model(images)
                for task in tasks:
                    valid_indices = gt_labels[task.name] != -100
                    if torch.any(valid_indices):
                        task_loss = criterions[task.name](
                            outputs[task.name][valid_indices],
                            gt_labels[task.name][valid_indices],
                        )
                        task_val_losses[task.name] += (
                            task_loss.item() * valid_indices.sum().item()
                        )
                        task_val_counts[task.name] += valid_indices.sum().item()
                        _, predicted = torch.max(outputs[task.name].data, 1)
                        all_preds[task.name].extend(
                            predicted[valid_indices].cpu().numpy()
                        )
                        all_labels[task.name].extend(
                            gt_labels[task.name][valid_indices].cpu().numpy()
                        )

    avg_task_val_losses = {
        name: (task_val_losses[name] / task_val_counts[name])
        if task_val_counts[name] > 0
        else 0.0
        for name in TASK_NAMES
    }

    static_val_weights = {task.name: task.weight for task in tasks}
    avg_val_loss = sum(
        avg_task_val_losses[name] * static_val_weights[name] for name in TASK_NAMES
    )
    accuracies = {
        name: accuracy_score(all_labels[name], all_preds[name])
        if all_labels[name]
        else 0.0
        for name in TASK_NAMES
    }
    avg_accuracy = np.mean(list(accuracies.values()))

    # --- Confusion Matrices ---
    for task in tasks:
        if all_labels[task.name]:
            cm = confusion_matrix(
                all_labels[task.name],
                all_preds[task.name],
                labels=range(len(task.class_labels)),
            )
            cm_normalized = confusion_matrix(
                all_labels[task.name], all_preds[task.name], normalize="true"
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=task.class_labels,
                yticklabels=task.class_labels,
            )
            (
                plt.title(f"{task.name} CM (Epoch {epoch + 1})"),
                plt.ylabel("Actual"),
                plt.xlabel("Predicted"),
            )
            (
                plt.savefig(cm_dir / task.name / f"cm_epoch_{epoch + 1}.png"),
                plt.close(),
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=task.class_labels,
                yticklabels=task.class_labels,
            )
            (
                plt.title(f"{task.name} CM (Epoch {epoch + 1})"),
                plt.ylabel("Actual"),
                plt.xlabel("Predicted"),
            )
            (
                plt.savefig(
                    cm_dir / task.name / f"cm_normalized_epoch_{epoch + 1}.png"
                ),
                plt.close(),
            )
            
    return avg_val_loss, avg_accuracy, avg_task_val_losses, accuracies


def main():
    """Main function to run the training and validation script."""
    args = get_args()
    set_seed(args.seed)
    device = args.device
    
    TASKS = setup_tasks(args)
    NUM_TASKS = len(TASKS)
    TASK_NAMES = [task.name for task in TASKS]
    TASK_NAME_TO_IDX = {"Age": 0, "Gender": 1, "Emotion": 2}

    output_dir, ckpt_dir, cm_dir = setup_directories(args, TASKS)
    
    model, transform = setup_model(args, TASKS, device)
    
    optimizer, scaler, scheduler, start_epoch = setup_optimizer_and_scheduler(args, model, device)
    
    train_loader, val_loader = setup_dataloaders(args, TASKS, transform, device)

    # --------------------------------------------------#
    #                   Loss functions                  #
    # --------------------------------------------------#
    criterions = {task.name: task.criterion(ignore_index=-100) for task in TASKS}

    # ----------------------------------------------------#
    #                   CSV logger setup                  #
    # ----------------------------------------------------#
    log_path = output_dir / "log.csv"
    log_header = ["epoch", "learning_rate", "avg_train_loss", "avg_val_loss"]
    log_header.extend([f"{task.name}_train_loss" for task in TASKS])
    log_header.extend([f"{task.name}_val_loss" for task in TASKS])
    log_header.extend([f"{task.name}_val_acc" for task in TASKS])
    if args.use_uw:
        log_header.extend([f"{task.name}_variance" for task in TASKS])
    if args.use_running_means:
        log_header.extend([f"{task.name}_weight" for task in TASKS])

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_header)
    print("CSV log file created at log.csv")
    print(f"Running training for the following tasks: {[task.name for task in TASKS]}")
    print_trainable_params(model)
    
    # --------------------------------------------------------------#
    #                   Training & Validation loop                  #
    # --------------------------------------------------------------#

    best_val_loss = float("inf")
    best_avg_acc = 0.0
    epochs_no_improve = 0
    running_means = (
        RunningMeans(task_names=TASK_NAMES, alpha=args.ema_alpha)
        if NUM_TASKS > 1
        else None
    )
    best_task_accuracies = {t: 0.0 for t in TASK_NAMES}
    
    for epoch in range(start_epoch, args.epochs):
        save_training_state(
            ckpt_dir / "training_state", optimizer, scheduler, scaler, epoch
        )
        
        avg_train_loss, avg_task_train_losses, task_weights = train_epoch(
            epoch, args, model, train_loader, device, optimizer, scaler, criterions, TASKS, TASK_NAME_TO_IDX, running_means
        )
        
        avg_val_loss, avg_accuracy, avg_task_val_losses, accuracies = validate_epoch(
            epoch, args, model, val_loader, device, criterions, TASKS, TASK_NAME_TO_IDX, cm_dir
        )

        # --- Scheduler Step ---
        if args.scheduler_type == "plateau":
            if args.use_avg_val_to_early_stop:
                scheduler.step(avg_val_loss)
            else:
                scheduler.step(avg_accuracy)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n--- Epoch {epoch + 1}/{args.epochs} Summary ---")
        print(
            f"  Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Avg Val Acc: {avg_accuracy:.4f}"
        )
        print(f"  Current LR: {current_lr:g}")
        for name in TASK_NAMES:
            print(
                f"    - {name}: Acc: {accuracies[name]:.4f}, Loss: {avg_task_val_losses[name]:.4f}"
            )

        try:
            message = f"Epoch {epoch + 1} MTL\n🔹Train Loss: {avg_train_loss:.4f}\n🔸Val Loss: {avg_val_loss:.4f}\n✅Avg Val Acc: {avg_accuracy:.3f}\n"
            for name in TASK_NAMES:
                message += f"  - {name}: {accuracies[name]:.3f}\n"
            # send_telegram_message(message)
        except Exception as e:
            print(f"Could not send Telegram message: {e}")

        # --- CSV Logging ---
        log_row = [
            epoch + 1,
            round(current_lr, 8),
            round(avg_train_loss, 5),
            round(avg_val_loss, 5),
        ]
        log_row.extend(
            [round(avg_task_train_losses.get(t.name, 0.0), 5) for t in TASKS]
        )
        log_row.extend([round(avg_task_val_losses.get(t.name, 0.0), 5) for t in TASKS])
        log_row.extend([round(accuracies.get(t.name, 0.0), 5) for t in TASKS])
        if args.use_uw:
            for v in model.log_vars:
                val = round(torch.exp(v).item(), 5)
                if val != 1:
                    log_row.append(val)
        if args.use_running_means:
            log_row.extend([round(task_weights.get(t.name, 0.0), 5) for t in TASKS])
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(log_row)

        # --- Early Stopping & Model Checkpointing ---
        is_best_overall = (
            args.use_avg_val_to_early_stop and avg_val_loss < best_val_loss
        ) or (args.use_avg_acc_to_early_stop and avg_accuracy > best_avg_acc)

        if is_best_overall:
            print(
                f"Overall performance improved. Best AvgLoss: {best_val_loss:.4f} -> {avg_val_loss:.4f}, Best AvgAcc: {best_avg_acc:.4f} -> {avg_accuracy:.4f}"
            )
            best_val_loss = min(avg_val_loss, best_val_loss)
            best_avg_acc = max(avg_accuracy, best_avg_acc)
            epochs_no_improve = 0
            save_path = ckpt_dir / "best_overall_model"
            print(f"Saving best overall model to {save_path}")
            if args.use_peft or args.use_mtlora:
                model.save_adapters_peft(ckpt_dir / f"model_{epoch + 1}.pt")
            else:
                model.save_trained(ckpt_dir / f"model_{epoch + 1}.pt")
        else:
            epochs_no_improve += 1
            print(
                f"Overall performance did not improve for {epochs_no_improve} epoch(s)."
            )

        for task_name in TASK_NAMES:
            if accuracies[task_name] > best_task_accuracies[task_name]:
                best_task_accuracies[task_name] = accuracies[task_name]
                print(
                    f"Best '{task_name}' accuracy improved to {accuracies[task_name]:.4f}."
                )
                save_path = ckpt_dir / f"best_{task_name}_model"
                print(f"Saving best '{task_name}' model to {save_path}")

                if args.use_peft or args.use_mtlora:
                    model.save_adapters_peft(ckpt_dir / f"model_{epoch + 1}.pt")
                else:
                    model.save_trained(ckpt_dir / f"model_{epoch + 1}.pt")

        if (
            args.use_early_stopping
            and epochs_no_improve >= args.early_stopping_patience
        ):
            print(
                f"\n--- Early stopping triggered after {epochs_no_improve} epochs with no improvement. ---"
            )
            break
        print("-" * 60)


if __name__ == "__main__":
    main()
