"""Contains Datasets and sampler classes"""

import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
import random
import math
import torchvision.transforms as transforms
from collections import Counter

# Task indices are used to access specific labels from the output tensor
AGE_IDX = 0
GENDER_IDX = 1
EMOTION_IDX = 2
ALL_TASK_IDXS = [AGE_IDX, GENDER_IDX, EMOTION_IDX]


def _gather_labels_fast(dataset):
    """Helper to efficiently get all labels from a dataset object."""
    ages = dataset._get_all_labels_for_task(AGE_IDX)
    genders = dataset._get_all_labels_for_task(GENDER_IDX)
    emotions = dataset._get_all_labels_for_task(EMOTION_IDX)
    return ages, genders, emotions


def build_weighted_sampler(
    dataset, class_weights_per_task, device, combine="mean", min_weight: float = 1e-4
):
    """
    Build a WeightedRandomSampler for a (possibly multi-task) dataset.

    Args:
        dataset: BaseDataset | MultiDataset | TaskBalanceDataset
        class_weights_per_task: list/tuple of Tensors [age_w, gender_w, emotion_w]
            Each tensor maps class index -> weight.
        combine: how to merge task weights per sample: 'mean' | 'sum' | 'max'
        min_weight: lower bound applied if a sample has no valid labels

    Returns:
        sampler, weights_tensor
    """
    assert (
        isinstance(class_weights_per_task, (list, tuple))
        and len(class_weights_per_task) >= 3
    ), "class_weights_per_task must be a list of 3 tensors [age, gender, emotion]"

    age_w, gender_w, emotion_w = class_weights_per_task[:3]

    # Ensure weights are on CPU for numpy compatibility if needed
    age_w = age_w.cpu() if age_w is not None else torch.tensor([])
    gender_w = gender_w.cpu() if gender_w is not None else torch.tensor([])
    emotion_w = emotion_w.cpu() if emotion_w is not None else torch.tensor([])

    ages, genders, emotions = _gather_labels_fast(dataset)
    n = len(ages)

    weights = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        w_parts = []
        a, g, e = ages[i], genders[i], emotions[i]

        # Safely access weights
        if a != -100 and int(a) < len(age_w):
            w_parts.append(age_w[int(a)].item())
        if g != -100 and int(g) < len(gender_w):
            w_parts.append(gender_w[int(g)].item())
        if e != -100 and int(e) < len(emotion_w):
            w_parts.append(emotion_w[int(e)].item())

        if not w_parts:
            weights[i] = min_weight
        else:
            if combine == "sum":
                weights[i] = sum(w_parts)
            elif combine == "max":
                weights[i] = max(w_parts)
            else:  # mean (default)
                weights[i] = sum(w_parts) / len(w_parts)

    # Normalize weights to have mean ~ 1.0 (keeps sampler numerically stable)
    mean_w = weights.mean().clamp_min(1e-8)
    weights = weights / mean_w
    weights = weights.to(device)
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    return sampler, weights


class WeightCalculationMixin:
    """
    A mixin class providing centralized logic for calculating class and task weights.
    """

    def get_class_weights(self, task_idx: int, weighting_method: str = "normalized"):
        if not hasattr(self, "class_weights_cache"):
            self.class_weights_cache = {}
        cache_key = (task_idx, weighting_method)
        if cache_key in self.class_weights_cache:
            return self.class_weights_cache[cache_key]

        if self.verbose:
            print(
                f"Computing class weights for task {task_idx} with method '{weighting_method}'"
            )

        all_task_data = self._get_all_labels_for_task(task_idx)
        class_counts = Counter(label for label in all_task_data if label != -100)
        total_valid_samples = sum(class_counts.values())

        if total_valid_samples == 0:
            if self.verbose:
                print(f"Warning: No valid samples found for task idx {task_idx}")
            return None

        class_indices = sorted(class_counts.keys())
        weights_array = []

        if weighting_method in ("default", "normalized"):
            num_classes = len(class_counts)
            for class_idx in class_indices:
                weight = total_valid_samples / (num_classes * class_counts[class_idx])
                weights_array.append(weight)
        elif weighting_method == "inverse_sqrt":
            for class_idx in class_indices:
                weights_array.append(1.0 / math.sqrt(max(class_counts[class_idx], 1)))
        else:
            raise ValueError(f"Unknown weighting_method: '{weighting_method}'.")

        if weighting_method == "normalized" and weights_array:
            max_weight = max(weights_array)
            if max_weight > 0:
                weights_array = [w / max_weight for w in weights_array]

        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
        self.class_weights_cache[cache_key] = weights_tensor

        if self.verbose:
            print(
                f"  Class distribution (task {task_idx}): {dict(sorted(class_counts.items()))}"
            )
            print(
                f"  Final class weights (task {task_idx}): {[round(w, 3) for w in weights_tensor.tolist()]}"
            )

        return weights_tensor


class BaseDataset(Dataset, WeightCalculationMixin):
    """
    Loads a single dataset, accessing labels via .iloc on the dataframe for efficiency.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        split="train",
        verbose=False,
        image_base_root: str = None,  # Accepts an explicit image root
        downsample_target: str = None,
        downsample_fraction: float = 1.0,
        downsample_seed=2025,
    ):
        self.verbose = verbose
        self.root = root  # This is the root of the LABEL file
        self.transform = transform
        self.split = split

        # Use the explicitly provided image root. This is the robust solution.
        if image_base_root:
            self.base_root = image_base_root
        else:
            # Fallback to the old, brittle logic if the new argument isn't provided
            print(
                "Warning: 'image_base_root' not provided. Using fallback logic to determine image root."
            )
            self.base_root = root.split("datasets_with_standard_labels")[0]

        labels_path = os.path.join(root, split, "labels.csv")
        self.data = pd.read_csv(labels_path)

        # Pre-process label columns in the dataframe
        self.data["Age"] = self.data["Age"].fillna(-100).astype(int)
        self.data["Gender"] = self.data["Gender"].fillna(-100).astype(int)
        self.data["Facial Emotion"] = (
            self.data["Facial Emotion"].fillna(-100).astype(int)
        )

        # Store integer indices of label columns for fast .iloc access
        self.age_col_idx = self.data.columns.get_loc("Age")
        self.gender_col_idx = self.data.columns.get_loc("Gender")
        self.emotion_col_idx = self.data.columns.get_loc("Facial Emotion")

        if (
            split == "train"
            and downsample_target
            and downsample_target.lower() in root.lower()
            and 0 < downsample_fraction < 1
        ):
            self._stratified_downsample(downsample_fraction, downsample_seed)

        # Correctly joins the image base root with the relative path from the CSV.
        self.img_paths = [
            os.path.join(self.base_root, path) for path in self.data["Path"]
        ]

    def _stratified_downsample(self, fraction: float, seed: int):
        original_size = len(self.data)
        num_to_remove = original_size - int(math.ceil(fraction * original_size))
        if num_to_remove <= 0:
            return

        rng = np.random.RandomState(seed)
        age_counts = self.data["Age"].value_counts()
        eligible_classes = age_counts[age_counts > age_counts.mean()]
        if eligible_classes.sum() < num_to_remove:
            eligible_classes = age_counts

        probs = eligible_classes / eligible_classes.sum()
        removal_alloc = rng.multinomial(num_to_remove, probs)

        indices_to_drop = []
        for age_class, k in zip(eligible_classes.index, removal_alloc):
            if k == 0:
                continue
            class_indices = self.data.index[self.data["Age"] == age_class].to_numpy()
            indices_to_drop.append(
                rng.choice(
                    class_indices, size=min(k, len(class_indices)), replace=False
                )
            )

        if indices_to_drop:
            self.data = self.data.drop(
                index=np.concatenate(indices_to_drop)
            ).reset_index(drop=True)

        if self.verbose:
            print(
                f"[BaseDataset] Downsampled '{os.path.basename(self.root)}' from {original_size} to {len(self.data)} samples."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print("\n" + "=" * 80)
            print("FATAL ERROR: Image file not found at the constructed path.")
            print(f"  -> Failing Path: {img_path}")
            print("-" * 50)
            print("This path was constructed from two parts:")
            print(f"  1. The 'image_base_root': {self.base_root}")
            print(
                f"  2. The raw path from the CSV file (row {idx}): {self.data.iloc[idx]['Path']}"
            )
            print("-" * 50)
            print(
                "ACTION: Please verify that joining these two parts points to a valid image file."
            )
            print("=" * 80 + "\n")
            raise

        if self.transform:
            image = self.transform(image)

        age_label = self.data.iloc[idx, self.age_col_idx]
        gender_label = self.data.iloc[idx, self.gender_col_idx]
        emotion_label = self.data.iloc[idx, self.emotion_col_idx]

        labels = torch.tensor(
            [age_label, gender_label, emotion_label], dtype=torch.long
        )

        return image, labels

    def _get_col_idx(self, task_idx: int):
        if task_idx == AGE_IDX:
            return self.age_col_idx
        if task_idx == GENDER_IDX:
            return self.gender_col_idx
        if task_idx == EMOTION_IDX:
            return self.emotion_col_idx
        raise ValueError(f"Unknown task idx for column lookup: {task_idx}")

    def _get_all_labels_for_task(self, task_idx: int):
        if task_idx == AGE_IDX:
            return self.data["Age"].values
        if task_idx == GENDER_IDX:
            return self.data["Gender"].values
        if task_idx == EMOTION_IDX:
            return self.data["Facial Emotion"].values
        raise ValueError(f"Unknown task idx: {task_idx}")

    def _get_task_counts_and_total_len(self):
        task_counts = {
            AGE_IDX: (self.data["Age"] != -100).sum(),
            GENDER_IDX: (self.data["Gender"] != -100).sum(),
            EMOTION_IDX: (self.data["Facial Emotion"] != -100).sum(),
        }
        return task_counts, len(self)


class MultiDataset(Dataset, WeightCalculationMixin):
    """
    Aggregates multiple BaseDataset instances into a single, unified dataset.
    """

    def __init__(
        self,
        dataset_names,
        transform=None,
        split="train",
        datasets_root="datasets_with_standard_labels",
        all_datasets=False,
        verbose=False,
        image_base_root: str = None,
    ):
        self.transform = transform
        self.split = split
        self.datasets_root = datasets_root
        self.verbose = verbose

        if all_datasets:
            dataset_names = [
                d
                for d in os.listdir(datasets_root)
                if os.path.isdir(os.path.join(datasets_root, d))
            ]
            if self.verbose:
                print(f"Loading all available datasets: {dataset_names}")

        self.datasets = []
        self.cumulative_lengths = [0]

        for name in dataset_names:
            path = os.path.join(datasets_root, name)
            if os.path.exists(os.path.join(path, split)):
                try:
                    dataset = BaseDataset(
                        root=path,
                        transform=transform,
                        split=split,
                        verbose=verbose,
                        image_base_root=image_base_root,
                    )
                    self.datasets.append(dataset)
                    self.cumulative_lengths.append(
                        self.cumulative_lengths[-1] + len(dataset)
                    )
                    if self.verbose:
                        print(f"✓ Loaded {len(dataset)} samples from {name}/{split}")
                except Exception as e:
                    print(f"✗ Warning: Could not load dataset {name}: {e}")

        if not self.datasets:
            raise ValueError("No datasets could be loaded successfully.")
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if not 0 <= idx < self.total_length:
            raise IndexError(
                f"Index {idx} is out of range for MultiDataset of size {self.total_length}"
            )

        dataset_idx = next(
            i
            for i, total_len in enumerate(self.cumulative_lengths[1:])
            if idx < total_len
        )
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

    def _get_all_labels_for_task(self, task_idx: int):
        return np.concatenate(
            [d._get_all_labels_for_task(task_idx) for d in self.datasets]
        )

    def _get_task_counts_and_total_len(self):
        task_counts = {AGE_IDX: 0, GENDER_IDX: 0, EMOTION_IDX: 0}
        for dataset in self.datasets:
            counts, _ = dataset._get_task_counts_and_total_len()
            for idx in ALL_TASK_IDXS:
                task_counts[idx] += counts[idx]
        return task_counts, self.total_length


class TaskBalanceDataset(MultiDataset):
    """
    Extends MultiDataset to balance the dataset by oversampling samples for under-represented tasks.
    """

    def __init__(
        self,
        *args,
        balance_task: dict = None,
        augment_duplicate: transforms.Compose = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.balance_task = balance_task or {}
        self.augment_duplicate = augment_duplicate
        self._build_flattened_index()
        if self.balance_task:
            self._apply_task_balancing()

    def _build_flattened_index(self):
        self.index_map = [
            [ds_idx, loc_idx, False]
            for ds_idx, ds in enumerate(self.datasets)
            for loc_idx in range(len(ds))
        ]

    def _apply_task_balancing(self):
        original_len = len(self.index_map)
        if self.verbose:
            print("\nApplying task balancing...")
        for task_idx, desired_fraction in self.balance_task.items():
            valid_indices = [
                i
                for i, (ds_idx, loc_idx, _) in enumerate(self.index_map)
                if self.datasets[ds_idx].data.iloc[
                    loc_idx, self.datasets[ds_idx]._get_col_idx(task_idx)
                ]
                != -100
            ]
            current_count = len(valid_indices)
            current_fraction = current_count / original_len if original_len > 0 else 0
            if current_fraction >= desired_fraction:
                continue

            n_to_add = int(
                (desired_fraction * original_len - current_count)
                / (1 - desired_fraction)
            )
            if n_to_add <= 0:
                continue

            duplicated_indices = random.choices(valid_indices, k=n_to_add)
            for original_idx in duplicated_indices:
                ds_idx, loc_idx, _ = self.index_map[original_idx]
                self.index_map.append([ds_idx, loc_idx, True])

            if self.verbose:
                print(
                    f"[Task {task_idx}] Duplicated {n_to_add} samples. New fraction: {(current_count + n_to_add) / len(self.index_map):.1%}"
                )

        random.shuffle(self.index_map)
        if self.verbose:
            print(f"Final balanced dataset size: {len(self.index_map)}\n")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self.index_map):
            raise IndexError(
                f"Index {idx} is out of range for TaskBalanceDataset of size {len(self.index_map)}"
            )

        dataset_idx, local_idx, is_duplicated = self.index_map[idx]
        dataset = self.datasets[dataset_idx]

        if is_duplicated and self.augment_duplicate:
            img_path = dataset.img_paths[local_idx]
            image = Image.open(img_path).convert("RGB")
            image = self.augment_duplicate(image)
            age_label = dataset.data.iloc[local_idx, dataset.age_col_idx]
            gender_label = dataset.data.iloc[local_idx, dataset.gender_col_idx]
            emotion_label = dataset.data.iloc[local_idx, dataset.emotion_col_idx]
            labels = torch.tensor(
                [age_label, gender_label, emotion_label], dtype=torch.long
            )
            return image, labels
        else:
            return dataset[local_idx]

    def _get_all_labels_for_task(self, task_idx: int):
        return np.array(
            [
                self.datasets[ds_idx].data.iloc[
                    loc_idx, self.datasets[ds_idx]._get_col_idx(task_idx)
                ]
                for ds_idx, loc_idx, _ in self.index_map
            ]
        )

    def _get_task_counts_and_total_len(self):
        task_counts = {idx: 0 for idx in ALL_TASK_IDXS}
        for ds_idx, loc_idx, _ in self.index_map:
            dataset = self.datasets[ds_idx]
            if dataset.data.iloc[loc_idx, dataset.age_col_idx] != -100:
                task_counts[AGE_IDX] += 1
            if dataset.data.iloc[loc_idx, dataset.gender_col_idx] != -100:
                task_counts[GENDER_IDX] += 1
            if dataset.data.iloc[loc_idx, dataset.emotion_col_idx] != -100:
                task_counts[EMOTION_IDX] += 1
        return task_counts, len(self.index_map)
