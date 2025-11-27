"""Code to produce PCA graph (2 component only)"""

import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.decomposition import PCA

load_dotenv()

from utils.commons import *
from utils.dataset import MultiDataset
from src.model import *
from utils.task_config import Task

parser = argparse.ArgumentParser(description="Run pca visualization on model features.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=["Age", "Gender", "Emotion"],
    help="The task to visualize (e.g., 'Age', 'Gender', 'Emotion').",
)
parser.add_argument(
    "--weights",
    type=str,
    default="trained",
    choices=["trained", "untrained"],
    help="Whether to load 'trained' weights or use an 'untrained' model.",
)
parser.add_argument("--dataset", type=str)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Model checkpoints directory"
)
parser.add_argument(
    "--peft_type",
    type=str,
    default="lora",
    choices=["lora", "mtlora"],
    help="Whether to load lora or use mtlora.",
)
args = parser.parse_args()


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
TASKS = [task for task in ALL_POSSIBLE_TASKS]
backbone, transform, _ = get_backbone("PE-Core-L14-336", apply_migration=True)
res_block_layers = len(backbone.transformer.resblocks)
layer_cutoff_idx = res_block_layers - 99
model = MTLModel(
    backbone,
    tasks=TASKS,
    rank=64,
    truncate_idx=(res_block_layers - 2),
    last_lora_layers=layer_cutoff_idx,
    use_lora=True,
    use_mtl_lora=(args.peft_type == "mtlora"),
    use_mtl_attn_pool=(args.peft_type == "mtlora"),
    use_deep_head=True,
).to("cuda")
if args.weights == "trained":
    model.load_adapters_peft(args.checkpoint)

model.backbone.merge_and_unload()
model.eval()

labels_root = "./labels"
image_base_root = "."

test_dataset = MultiDataset(
    dataset_names=[args.dataset],
    transform=transform,
    split="test",
    datasets_root=labels_root,
    image_base_root=image_base_root,
    verbose=True,
)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=False
)


all_features = []
all_labels = []

idx = 0
if args.task == "Age":
    class_labels = [
        "0-2",
        "3-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+",
    ]
    idx = 0
elif args.task == "Gender":
    class_labels = ["Male", "Female"]
    idx = 1
elif args.task == "Emotion":
    class_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
    idx = 2


print("Extracting features from the test set...")
for images, labels in tqdm(test_loader):
    with torch.no_grad():
        images = images.to("cuda")
        labels = labels.to("cuda")
        labels = labels[:, idx]
        if args.peft_type == "mtlora":
            feat = model._forward_mtl_block(
                images, return_feat=True, feat_to_return=args.task
            )
            feat = feat.squeeze(1)
        else:
            feat = model.backbone(images)

        all_features.append(feat.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

print("\nFeature extraction complete.")


all_features_np = np.concatenate(all_features, axis=0)
all_labels_np = np.concatenate(all_labels, axis=0)

print(f"Total features shape: {all_features_np.shape}")
print(f"Total labels shape: {all_labels_np.shape}")

n_components_pca = 2
print(f"Calculating PCA (n_components={n_components_pca}) for visualization...")

pca = PCA(n_components=n_components_pca, random_state=42)

features_pca = pca.fit_transform(all_features_np)

print(f"PCA-transformed features shape: {features_pca.shape}")

print("Visualizing PCA and saving to 'pca_features_visualization.png'...")


palette = sns.color_palette("hls", len(class_labels))

mapped_labels = []
mapped_labels = [class_labels[i] for i in all_labels_np.astype(int)]


plt.figure(figsize=(14, 10))
sns.scatterplot(
    x=features_pca[:, 0],
    y=features_pca[:, 1],
    hue=mapped_labels,
    palette=palette,
    hue_order=class_labels,
    legend="full",
    alpha=0.6,
)


explained_variance_ratio = pca.explained_variance_ratio_
total_explained_variance = np.sum(explained_variance_ratio)


plt.title(f"PCA of Backbone Features ({args.dataset} Test Set)", fontsize=16)
plt.xlabel(
    f"Principal Component 1, explained var: {explained_variance_ratio[0]:.3f}",
    fontsize=12,
)
plt.ylabel(
    f"Principal Component 2, explained var: {explained_variance_ratio[1]:.3f}",
    fontsize=12,
)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plot_filename = f"pca_{args.task}_{args.weights}_{args.dataset}_{args.peft_type}.png"
plt.savefig(plot_filename, bbox_inches="tight")
plt.close()
print(f"PCA visualization saved successfully to {plot_filename}")


print("\nPCA Explained Variance:")
print(
    f"  - Principal Component 1: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0] * 100:.2f}%)"
)
print(
    f"  - Principal Component 2: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1] * 100:.2f}%)"
)
print(
    f"  - Total Variance Explained: {total_explained_variance:.4f} ({total_explained_variance * 100:.2f}%)"
)
