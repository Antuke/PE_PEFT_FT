"""
Code for testing. It parses a series of argument and instanciate the model, its wrapper, the dataset classes
and starts the testing loop.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load environment variables and set up paths
load_dotenv()

from utils.commons import *
from src.utils.config_parser import parse_config
from utils.dataset import MultiDataset
from src.model import *
from utils.task_config import Task


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
    parser = argparse.ArgumentParser(description="Multi-Task Learning Testing Script")

    # Paths and environment
    parser.add_argument(
        "--repo_path",
        type=str,
        default=os.getenv("REPO_PATH"),
        help="Path to the repository",
    )
    parser.add_argument(
        "--image_base_root",
        type=str,
        default="/user/asessa/dataset tesi/",
        help="Base root for images",
    )
    parser.add_argument(
        "--labels_root",
        type=str,
        default="/user/asessa/dataset tesi/LABELS",
        help="Root directory for labels",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./testing_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the model checkpoint directory",
    )

    # Testing configuration
    parser.add_argument(
        "--name", type=str, default="testing", help="Name of the testing run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use for testing (e.g., "cuda", "cpu")',
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Model configuration
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="PE-Core-L14-336",
        help="Name of the backbone model",
    )
    parser.add_argument("--rank", type=int, default=64, help="Rank for LoRA")
    parser.add_argument(
        "--use_mtlora",
        action="store_true",
        help="Use MT-LoRA architecture in the model",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Use PEFT LoRA architecture in the model",
    )
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=99,
        help="Number of layers to unfreeze (for model init)",
    )
    parser.add_argument(
        "--use_deep_head",
        action="store_true",
        help="Use a prediction head with one hidden layer",
    )
    parser.add_argument(
        "--load_trained",
        action="store_true",
        help="To load checkpoint in a different manner",
    )
    parser.add_argument("--compile_model", action="store_true", help="compile model")

    # Dataset and Tasks
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["RAF-DB"],
        help="List of dataset names to test on",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Emotion"],
        choices=["Age", "Gender", "Emotion"],
        help="Specify which tasks to evaluate. Example: --tasks Age Gender",
    )

    parser.add_argument(
        "--setup_from_json",
        type=str,
        help="Path to a config.json file to setup model configuration",
    )

    args = parse_config(parser)

    if args.setup_from_json:
        with open(args.setup_from_json, "r") as f:
            config = json.load(f)

        # Update model configuration from JSON
        if "backbone_name" in config:
            args.backbone_name = config["backbone_name"]
        if "rank" in config:
            args.rank = config["rank"]
        if "use_mtlora" in config:
            args.use_mtlora = config["use_mtlora"]
        if "use_peft" in config:
            args.use_peft = config["use_peft"]
        if "unfreeze_layers" in config:
            args.unfreeze_layers = config["unfreeze_layers"]
        if "use_deep_head" in config:
            args.use_deep_head = config["use_deep_head"]

            

        # Handle tasks: config.json has list of dicts, args.tasks expects list of strings
        if "tasks" in config:
            if (
                isinstance(config["tasks"], list)
                and len(config["tasks"]) > 0
                and isinstance(config["tasks"][0], dict)
            ):
                args.tasks = [t["name"] for t in config["tasks"]]
            elif (
                isinstance(config["tasks"], list)
                and len(config["tasks"]) > 0
                and isinstance(config["tasks"][0], str)
            ):
                args.tasks = config["tasks"]

    if not args.ckpt_path:
        parser.error("the following arguments are required: --ckpt_path")

    return args


def test_model(
    model,
    dataset_name,
    transform,
    output_dir,
    tasks,
    task_name_to_idx,
    device,
    batch_size,
    num_workers,
    labels_root,
    image_base_root,
):
    """
    Tests the model on a given dataset and saves the results.
    """
    print(f"\n--- Testing on dataset: {dataset_name} ---")

    test_dataset = MultiDataset(
        dataset_names=[dataset_name],
        transform=transform,
        split="test",
        datasets_root=labels_root,
        image_base_root=image_base_root,
        verbose=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    all_preds = {task.name: [] for task in tasks}
    all_labels = {task.name: [] for task in tasks}
    task_names = [task.name for task in tasks]

    with torch.no_grad():
        progress_bar_test = tqdm(test_loader, desc=f"Testing on {dataset_name}")
        for images, labels in progress_bar_test:
            images, labels = images.to(device), labels.to(device)
            gt_labels = {
                task.name: labels[:, task_name_to_idx[task.name]] for task in tasks
            }

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=False):
                outputs = model(images)
                for task in tasks:
                    task_name = task.name
                    valid_indices = gt_labels[task_name] != -100

                    if torch.any(valid_indices):
                        _, predicted = torch.max(outputs[task_name].data, 1)
                        all_preds[task_name].extend(
                            predicted[valid_indices].cpu().numpy()
                        )
                        all_labels[task_name].extend(
                            gt_labels[task_name][valid_indices].cpu().numpy()
                        )

    metrics = {}
    for task_name in task_names:
        if all_labels[task_name]:
            acc = accuracy_score(all_labels[task_name], all_preds[task_name])
            bal_acc = balanced_accuracy_score(
                all_labels[task_name], all_preds[task_name]
            )
            metrics[task_name] = {"accuracy": acc, "balanced_accuracy": bal_acc}
        else:
            metrics[task_name] = {"accuracy": 0.0, "balanced_accuracy": 0.0}

    print(f"\n--- Results for {dataset_name} ---")
    for task_name, task_metrics in metrics.items():
        print(f"  - {task_name} Accuracy: {task_metrics['accuracy']:.4f}")
        print(
            f"  - {task_name} Balanced Accuracy: {task_metrics['balanced_accuracy']:.4f}"
        )

    # Save confusion matrices
    cm_dir = output_dir / "confusion_matrices"
    dataset_cm_dir = cm_dir / dataset_name
    dataset_cm_dir.mkdir(exist_ok=True, parents=True)

    for task in tasks:
        task_name = task.name
        if all_labels[task_name]:
            cm_non_normalized = confusion_matrix(
                all_labels[task_name], all_preds[task_name]
            )
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_non_normalized,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=task.class_labels,
                yticklabels=task.class_labels,
            )
            plt.title(f"{task_name} Confusion Matrix (Sample Counts) on {dataset_name}")
            plt.ylabel("Actual"), plt.xlabel("Predicted")
            plt.savefig(dataset_cm_dir / f"cm_non_normalized_{task_name}.png")
            plt.close()

            cm_normalized = confusion_matrix(
                all_labels[task_name], all_preds[task_name], normalize="true"
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
            plt.title(f"{task_name} Normalized Confusion Matrix on {dataset_name}")
            plt.ylabel("Actual"), plt.xlabel("Predicted")
            plt.savefig(dataset_cm_dir / f"cm_normalized_{task_name}.png")
            plt.close()

    return metrics


def main():
    """Main function to run the testing script."""
    args = get_args()
    set_seed(args.seed)

    if args.repo_path:
        sys.path.append(args.repo_path)

    # --------------------------------------------------#
    #                   Configuration                   #
    # --------------------------------------------------#
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
        ),
        Task(
            name="Gender",
            class_labels=["Male", "Female"],
            criterion=torch.nn.CrossEntropyLoss,
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
        ),
    ]
    TASKS = [task for task in ALL_POSSIBLE_TASKS if task.name in args.tasks]
    if not TASKS:
        raise ValueError(
            "No valid tasks selected. Please choose from 'Age', 'Gender', 'Emotion'."
        )

    print(f"Running testing for the following tasks: {[task.name for task in TASKS]}")

    TASK_NAME_TO_IDX = {"Age": 0, "Gender": 1, "Emotion": 2}

    # -----------------------------------------------------------#
    #                   Output & Logging Setup                   #
    # -----------------------------------------------------------#
    run_name = f"run_{args.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    # ------------------------------------------------#
    #                   Model setup                   #
    # ------------------------------------------------#
    backbone, transform, _ = get_backbone(args.backbone_name, apply_migration=True)
    res_block_layers = len(backbone.transformer.resblocks)
    layer_cutoff_idx = res_block_layers - args.unfreeze_layers
    model = MTLModel(
        backbone,
        tasks=TASKS,
        rank=args.rank,
        truncate_idx=(res_block_layers - 2),
        last_lora_layers=layer_cutoff_idx,
        use_lora=args.use_peft,
        use_mtl_lora=args.use_mtlora,
        use_mtl_attn_pool=args.use_mtlora,
        use_deep_head=args.use_deep_head,
    ).to(args.device)

    if args.use_peft or args.use_mtlora:
        model.load_adapters_peft(args.ckpt_path)
        model.backbone.merge_and_unload()
    elif "legacy" in args.ckpt_path:
        model.load_trained_legacy(args.ckpt_path)
    else:
        model.load_trained(args.ckpt_path)
    print(f"Model loaded from checkpoint: {args.ckpt_path}")

    if args.compile_model:
        model = torch.compile(model)

    # --------------------------------------------------------------#
    #                       Testing Loop                           #
    # --------------------------------------------------------------#
    all_results = {"ckpt_path": args.ckpt_path}
    for dataset_name in args.dataset_names:
        results = test_model(
            model=model,
            dataset_name=dataset_name,
            transform=transform,
            output_dir=output_dir,
            tasks=TASKS,
            task_name_to_idx=TASK_NAME_TO_IDX,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            labels_root=args.labels_root,
            image_base_root=args.image_base_root,
        )
        all_results[dataset_name] = results

    # Save all results to a single JSON file
    results_file = output_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll testing results saved to: {results_file}")


if __name__ == "__main__":
    main()
