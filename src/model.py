from einops import rearrange
from torch.nn import functional as F
from dotenv import load_dotenv
import os
import sys

load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from core.vision_encoder.pe import SelfAttention, AttentionPooling
import torch.nn as nn
from typing import Dict, List
from utils.task_config import Task
import torch
from typing import Optional, Union, Mapping, OrderedDict
from src.dlora import *
from peft import PeftModel, get_peft_model, LoraConfig

DROPOUT_P = 0.5
MAX_TASKS = 3

class MTLModel(nn.Module):
    def __init__(
        self,
        backbone,
        tasks: List[Task],
        rank: int = 64,
        use_lora: bool = True,
        truncate_idx: int = 22,
        last_lora_layers: int = 99,
        lora_dropout: float = 0.5,
        use_mtl_lora: bool = False,
        use_deep_head: bool = False,
        use_batch_norm: bool = True,
        use_mtl_attn_pool: bool = True,
        use_dora: bool = True,
    ):
        super().__init__()
        self.use_mtl_attn_pool = use_mtl_attn_pool
        self.tasks = tasks
        self.use_mtl_lora = use_mtl_lora
        self.use_deep_head = use_deep_head
        self.use_lora = use_lora
        self.use_mtlora = use_mtl_lora
        
        # log_vars is for uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(MAX_TASKS))
        self.backbone = backbone

        if self.use_mtl_lora:
            self._setup_mtl_lora(truncate_idx, rank, lora_dropout)

        if use_lora:
            self._setup_peft_lora(rank, last_lora_layers, use_dora, lora_dropout)

        self._setup_prediction_heads(use_batch_norm)

        self.backbone.del_muda()

    def _setup_mtl_lora(self, truncate_idx, rank, lora_dropout):
        task_names = [task.name for task in self.tasks]
        width = self.backbone.width
        heads = self.backbone.heads
        rope = self.backbone.rope

        # save last residual attention block, as we need the weights values to seed the new mtl version
        orig_last_block = self.backbone.transformer.resblocks[-1]
        self.ln_post = self.backbone.ln_post

        # save the attention pooling, as we need the weights values to seed the task specifics attention pooling layers
        orig_attn_pool = self.backbone.attn_pool.to("cuda")

        self.backbone.truncate(
            layer_idx=truncate_idx
        )  # 23th block becomes the last (the idx is 22)

        # mtl block that produces t-task specific features maps, plus a shared one
        self.mtl_layer = MTLoRAResidualAttentionBlock(
            d_model=width,
            n_head=heads,
            rope=rope,
            r={"shared": rank, **{name: rank for name in task_names}},
            tasks=task_names,
            shared_mode="matrix",
            lora_shared_scale=0.0,  # We do not use the shared matrix, so we set it's scale to 0
        )

        self.mtl_layer.load_from_original_block(orig_last_block)
        print(
            "MTL-LoRA final block created and initialized from pretrained weights."
        )

        if self.use_mtl_attn_pool:
            self.attn_pool = MTLoRAAttentionPooling(
                embed_dim=width,
                num_heads=8,
                tasks=task_names,
                r={"shared": rank, **{name: rank for name in task_names}},
                lora_dropout=lora_dropout,
                lora_task_scale=1.0,
                lora_shared_scale=0.0,
            )
            self.attn_pool.load_from_original(orig_attn_pool)
        else:
            self.task_specific_attn_pool = nn.ModuleDict(
                {
                    task.name: AttentionPooling(embed_dim=width, num_heads=8)
                    for task in self.tasks
                }
            )
            for task in self.tasks:
                self.task_specific_attn_pool[task.name].load_state_dict(
                    orig_attn_pool.state_dict()
                )
            print("Task-specific Attention Pooling layers created and initialized.")

        del self.backbone.attn_pool

    def _setup_peft_lora(self, rank, last_lora_layers, use_dora, lora_dropout):
        # You can modify this list if you want to target only attention layers or mlp layers
        target_layers = ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
        target_modules = []
        for name, param in self.backbone.named_modules():
            if not isinstance(param, nn.Linear):
                continue
            is_target_layer = any(s in name for s in target_layers)
            if is_target_layer:
                if "attn_pool" in name:
                    target_modules.append(name)
                elif "transformer.resblocks" in name:
                    layer_idx = int(name.split(".")[2])
                    if layer_idx >= last_lora_layers:
                        target_modules.append(name)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=target_modules,
            use_dora=use_dora,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        print("PEFT LoRA module added")

    def _setup_prediction_heads(self, use_batch_norm):
        if self.use_deep_head == False:
            self.prediction_layers = nn.ModuleDict(
                {
                    task.name: nn.Sequential(
                        nn.BatchNorm1d(self.backbone.output_dim)
                        if use_batch_norm
                        else nn.Identity(),
                        nn.Dropout(p=DROPOUT_P),
                        nn.Linear(self.backbone.output_dim, len(task.class_labels)),
                    )
                    for task in self.tasks
                }
            )
            print("Task-specific prediction heads created.")
        else:
            self.prediction_layers = nn.ModuleDict(
                {
                    task.name: nn.Sequential(
                        nn.BatchNorm1d(self.backbone.output_dim)
                        if use_batch_norm
                        else nn.Identity(),
                        nn.Dropout(p=DROPOUT_P),
                        nn.Linear(self.backbone.output_dim, self.backbone.output_dim),
                        nn.GELU(),
                        nn.Linear(self.backbone.output_dim, len(task.class_labels)),
                    )
                    for task in self.tasks
                }
            )
            print("Task-specific prediction deep-heads created.")

    def enable_gradient_checkpointing(self):
        """Call this method after setting up parameter requires_grad"""
        backbone_has_trainable = any(
            param.requires_grad for param in self.backbone.parameters()
        )
        if backbone_has_trainable:
            self.backbone.set_grad_checkpointing()
            print(
                "Gradient checkpointing enabled for backbone (has trainable parameters)"
            )
        else:
            print(
                "Gradient checkpointing not enabled - backbone has no trainable parameters"
            )

    def forward(self, x: torch.Tensor):
        if self.use_mtl_lora:
            return self._forward_mtl_block(x)
        else:
            return self._forward_shared(x)

    def _forward_shared(self, x: torch.Tensor):
        logits = {}


        features = self.backbone(x)
        # print(features.shape)
        for task in self.tasks:
            logits[task.name] = self.prediction_layers[task.name](features)

        return logits

    def _forward_mtl_block(
        self, x: torch.Tensor, return_feat=False, feat_to_return="None"
    ):
        # Shared feature map from the backbone
        # norm=False, because normalization is "trained" on the feature map of the output of the last ResidualAttentionBlock
        # so we will normalize the task specific feature map, instead of the shared one
        # strip_cls_token=False, because in the PE paper it has been shown to be beneficial to keep it
        features = self.backbone.forward_features(x, norm=False, strip_cls_token=False)

        # Equal for each task, as our mtl layer follows a task-agnostic layer
        task_features_input = {task.name: features for task in self.tasks}

        # Returns also a shared features map, that is discarded,
        # task features is a dictionary, the key is task name, and the value is a tensor of shape (batch_size, n_tokens, d_model)
        # rappresting the task specific features map
        _, task_features = self.mtl_layer(features, x_tasks=task_features_input)

        normalized_task_features = {
            task.name: self.ln_post(task_features[task.name]) for task in self.tasks
        }

        if self.use_mtl_attn_pool:
            pooled_features = self.attn_pool(normalized_task_features)
        else:
            pooled_features = {}
            for task in self.tasks:
                feat = normalized_task_features[task.name]
                pooled_features[task.name] = self.task_specific_attn_pool[task.name](
                    feat
                )

        # this stuff is for pca/tsne visualization
        if return_feat:
            if feat_to_return == "Age":
                return pooled_features["Age"]
            elif feat_to_return == "Emotion":
                return pooled_features["Emotion"]
            elif feat_to_return == "Gender":
                return pooled_features["Gender"]

        logits = {}
        for task in self.tasks:
            # Squeeze the pooling dimension (1)
            pooled_feat = pooled_features[task.name].squeeze(
                1
            )  # (batch, 1, d_model) -> (batch, d_model)
            logits[task.name] = self.prediction_layers[task.name](pooled_feat)

        return logits

    def save_whole_model(self, filepath: str):
        print(f"Saving model state_dict to {filepath}")
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        state_dict = torch.load(filepath, map_location=device)
        self.load_state_dict(state_dict, strict=True)

    def save_adapters_peft(self, save_directory: str):
        print(f"Saving adapters to directory: {save_directory}")
        os.makedirs(save_directory, exist_ok=True)

        custom_layers_state_dict = {
            "prediction_layers": self.prediction_layers.state_dict()
        }

        if self.use_lora:
            self.backbone.save_pretrained(save_directory)

        if self.use_mtlora:
            custom_layers_state_dict["mtl_layer"] = self.mtl_layer.state_dict()
            # custom_layers_state_dict['task_specific_attn_pooling'] = self.task_specific_attn_pool.state_dict()
            custom_layers_state_dict["mtl_attn_pool"] = self.attn_pool.state_dict()

        torch.save(
            custom_layers_state_dict, os.path.join(save_directory, "custom_layers.pt")
        )
        print("Successfully saved PEFT backbone and custom task heads.")

    def load_heads(self, filepaths: List[str], device="cuda"):
        for ckpt in filepaths:
            checkpoint = torch.load(ckpt, map_location=device)
            model_state_dict = self.state_dict()

            if "prediction_layers" in checkpoint:
                for loaded_key, value in checkpoint["prediction_layers"].items():
                    new_key = loaded_key

                    # Remap prefix: 'heads.emotion.' -> 'prediction_layers.Emotion.'
                    if new_key.startswith("heads.emotion."):
                        new_key = new_key.replace(
                            "heads.emotion.", "prediction_layers.Emotion."
                        )

                    if new_key.startswith("heads.age."):
                        new_key = new_key.replace(
                            "heads.age.", "prediction_layers.Age."
                        )

                    if new_key.startswith("heads.gender."):
                        new_key = new_key.replace(
                            "heads.gender.", "prediction_layers.Gender."
                        )

                    # Remap final layer index for deep head: '.5.' -> '.4.'
                    if ".5." in new_key:
                        new_key = new_key.replace(".5.", ".4.")

                    if new_key in model_state_dict:
                        if model_state_dict[new_key].shape == value.shape:
                            model_state_dict[new_key].copy_(value)

    def load_adapters_peft(
        self, load_directory: str, custom_head_name: str = "custom_layers.pt"
    ):
        print(f"Loading adapters from directory: {load_directory}")
        if self.use_lora:
            self.backbone = self.backbone.merge_and_unload()
            self.backbone = PeftModel.from_pretrained(self.backbone, load_directory)

        custom_layers_path = os.path.join(load_directory, custom_head_name)
        if not os.path.exists(custom_layers_path):
            raise FileNotFoundError(
                f"Custom task heads file not found at {custom_layers_path}"
            )

        checkpoint = torch.load(
            custom_layers_path,
            map_location=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.prediction_layers.load_state_dict(checkpoint["prediction_layers"])

        if self.use_mtlora:
            try:
                self.mtl_layer.load_state_dict(checkpoint["mtl_layer"][0])
            except KeyError:
                self.mtl_layer.load_state_dict(checkpoint["mtl_layer"])
            self.attn_pool.load_state_dict(checkpoint["mtl_attn_pool"])

        print("Successfully loaded PEFT backbone and custom task heads.")

    def save_trained(self, filepath: str):
        trainable_param_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        trainable_module_paths = {
            ".".join(name.split(".")[:-1]) for name in trainable_param_names
        }

        state_to_save = {}
        full_state_dict = self.state_dict()

        for key, value in full_state_dict.items():
            if key in trainable_param_names:
                state_to_save[key] = value
                continue

            current_module_path = ".".join(key.split(".")[:-1])
            if current_module_path in trainable_module_paths:
                state_to_save[key] = value

        print(
            f"Saving {len(state_to_save)} state entries (parameters and buffers) to {filepath}"
        )
        torch.save(state_to_save, filepath)

    def load_trained(self, filepath: str):
        print(f"Loading trained state from {filepath}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(filepath, map_location=device)
        
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded {len(state_dict)} keys:")
        for k in state_dict.keys():
            print(f"  {k}")

        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} (Expected since we only load trained params)")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    def load_trained_legacy(self, filepath: str, device="cuda"):
        """The training of some checkpoint where done with a different model class,
        so there is the need of remapping the key names, so they match with this new model class"""
        print(f"Loading trained states from structured checkpoint: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)

        model_state_dict = self.state_dict()

        loaded_keys_count = 0
        skipped_keys = []
        remapped_keys_examples = {}

        if "backbone_state_dict" in checkpoint:
            print("\n--- Processing Backbone Weights ---")
            for loaded_key, value in checkpoint["backbone_state_dict"].items():
                new_key = loaded_key

                if new_key.startswith("strategy.backbone."):
                    new_key = new_key.replace("strategy.backbone.", "backbone.")

                if (
                    "attn.in_proj_weight" in new_key
                    and "attn.in_proj.weight" not in new_key
                ):
                    new_key = new_key.replace(
                        "attn.in_proj_weight", "attn.in_proj.weight"
                    )
                if (
                    "attn.in_proj_bias" in new_key
                    and "attn.in_proj.bias" not in new_key
                ):
                    new_key = new_key.replace("attn.in_proj_bias", "attn.in_proj.bias")

                if new_key in model_state_dict:
                    if model_state_dict[new_key].shape == value.shape:
                        model_state_dict[new_key].copy_(value)
                        loaded_keys_count += 1
                        if loaded_key != new_key and len(remapped_keys_examples) < 5:
                            remapped_keys_examples[loaded_key] = new_key
                    else:
                        skipped_keys.append(
                            f"{loaded_key} (Shape Mismatch: Model {model_state_dict[new_key].shape} vs Ckpt {value.shape})"
                        )
                else:
                    skipped_keys.append(
                        f"{loaded_key} (as {new_key}) -> Not found in model"
                    )

        if "prediction_layers" in checkpoint:
            print("\n--- Processing Prediction Head Weights ---")
            for loaded_key, value in checkpoint["prediction_layers"].items():
                new_key = loaded_key

                if new_key.startswith("heads.emotion."):
                    new_key = new_key.replace(
                        "heads.emotion.", "prediction_layers.Emotion."
                    )

                if new_key.startswith("heads.age."):
                    new_key = new_key.replace("heads.age.", "prediction_layers.Age.")

                if new_key.startswith("heads.gender."):
                    new_key = new_key.replace(
                        "heads.gender.", "prediction_layers.Gender."
                    )

                if ".5." in new_key:
                    new_key = new_key.replace(".5.", ".4.")

                # Validate, load, and update trackers
                if new_key in model_state_dict:
                    if model_state_dict[new_key].shape == value.shape:
                        model_state_dict[new_key].copy_(value)
                        loaded_keys_count += 1
                        if loaded_key != new_key and len(remapped_keys_examples) < 10:
                            remapped_keys_examples[loaded_key] = new_key
                    else:
                        skipped_keys.append(
                            f"{loaded_key} (Shape Mismatch: Model {model_state_dict[new_key].shape} vs Ckpt {value.shape})"
                        )
                else:
                    skipped_keys.append(
                        f"{loaded_key} (as {new_key}) -> Not found in model"
                    )

        if "attn_pool" in checkpoint:
            print("\n--- Processing Attention Pool Weights ---")
            for loaded_key, value in checkpoint["attn_pool"].items():
                # The attn_pool keys in the source file also have the 'strategy.backbone' prefix
                new_key = loaded_key.replace(
                    "strategy.backbone.attn_pool.", "backbone.attn_pool."
                )

                # Validate, load, and update trackers
                if new_key in model_state_dict:
                    if model_state_dict[new_key].shape == value.shape:
                        model_state_dict[new_key].copy_(value)
                        loaded_keys_count += 1
                        if loaded_key != new_key and len(remapped_keys_examples) < 15:
                            remapped_keys_examples[loaded_key] = new_key
                    else:
                        skipped_keys.append(
                            f"{loaded_key} (Shape Mismatch: Model {model_state_dict[new_key].shape} vs Ckpt {value.shape})"
                        )
                else:
                    skipped_keys.append(
                        f"{loaded_key} (as {new_key}) -> Not found in model"
                    )

        if loaded_keys_count == 0:
            print("LAODED 0")
            self.load_state_dict(
                torch.load(filepath, map_location=device), strict=False
            )


class MTLoRAResidualAttentionBlock(nn.Module):
    """Adaptation of Perception Encoder ResidualAttentionBlock with MTLora, to produce t-task specific feature-maps and a shared feature map"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = DROPOUT_P,
        tasks=None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode: str = "matrix",
    ):
        super().__init__()
        self.tasks = tasks
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim**-0.5
        self.rope = rope

        task_scales = {t: lora_task_scale for t in tasks}

        # MultiTask Lora for QKV matrices
        # (MTLoRAQKV does not actually compute attention, but returns the shared QKV matrices and the task-specific QKV matrices)
        self.attn = MTLoRAQKV(
            in_features=d_model,
            out_features=d_model,
            r=r,
            lora_shared_scale=lora_shared_scale,
            lora_task_scale=task_scales,
            lora_dropout=lora_dropout,
            tasks=tasks,
            trainable_scale_shared=trainable_scale_shared,
            trainable_scale_per_task=trainable_scale_per_task,
            shared_mode=shared_mode,
        )

        print(r)
        # MultiTask Lora for projection matrices in mha
        self.out_proj = MTLoRALinear(
            in_features=d_model,
            out_features=d_model,
            r=r,
            lora_shared_scale=lora_shared_scale,
            lora_task_scale=task_scales,
            lora_dropout=lora_dropout,
            tasks=tasks,
            trainable_scale_shared=trainable_scale_shared,
            trainable_scale_per_task=trainable_scale_per_task,
            shared_mode=shared_mode,
        )

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # LoRA-enabled MLP
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c_fc",
                        MTLoRALinear(
                            d_model,
                            mlp_width,
                            r=r,
                            lora_shared_scale=lora_shared_scale,
                            lora_task_scale=task_scales,
                            lora_dropout=lora_dropout,
                            tasks=tasks,
                            trainable_scale_shared=trainable_scale_shared,
                            trainable_scale_per_task=trainable_scale_per_task,
                            shared_mode=shared_mode,
                        ),
                    ),
                    ("gelu", act_layer()),
                    (
                        "c_proj",
                        MTLoRALinear(
                            mlp_width,
                            d_model,
                            r=r,
                            lora_shared_scale=lora_shared_scale,
                            lora_task_scale=task_scales,
                            lora_dropout=lora_dropout,
                            tasks=tasks,
                            trainable_scale_shared=trainable_scale_shared,
                            trainable_scale_per_task=trainable_scale_per_task,
                            shared_mode=shared_mode,
                        ),
                    ),
                ]
            )
        )

    def _call_attn(
        self,
        x_shared: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        x_tasks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # s is the number of patches/tokens, sequence length
        proj, proj_tasks = self.attn(
            x_shared, x_tasks
        )  # proj is (b s 3*d_model), proj_tasks is dict of (b s 3*d_model), one entry per task

        def compute_attention(projection_tensor):
            # Reshape Q, K, V
            # projection_tensor is (b s 3*d_model), need to split and rearrange
            _, s, _ = projection_tensor.shape
            # output_features from MTLoRAQKV is d_model, so 3 * d_model
            split_size = self.attn.q.linear.out_features  # This should be d_model

            # Unflatten into (b s 3 d_model) then transpose to get (3 b s d_model)
            q, k, v = (
                projection_tensor.unflatten(-1, (3, split_size))
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            # Rearrange for multi-head attention (b h s d)
            q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
            k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
            v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

            if self.rope:
                q, k = self.rope(q, k)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=self.scale
            )
            return rearrange(attn_output, "b h s d -> b s (h d)")

        # Process shared path
        attn_result = compute_attention(proj)

        # Process task-specific paths
        attn_tasks_results = {}
        if proj_tasks:
            for task, task_proj in proj_tasks.items():
                attn_tasks_results[task] = compute_attention(task_proj)

        # Apply output projection
        # out_proj is an MTLoRALinear, so its forward expects (x, x_tasks)
        shared_out, tasks_out = self.out_proj(
            attn_result, x_tasks=attn_tasks_results if attn_tasks_results else None
        )

        return shared_out, tasks_out

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        x_tasks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Attention block
        norm_x = self.ln_1(x)
        norm_x_tasks = (
            {task: self.ln_1(x_tasks[task]) for task in self.tasks} if x_tasks else None
        )

        attn_out, attn_tasks_out = self._call_attn(
            norm_x, attn_mask=attn_mask, x_tasks=norm_x_tasks
        )

        x = x + self.drop_path1(self.ls_1(attn_out))
        if attn_tasks_out and x_tasks:
            for task in self.tasks:
                x_tasks[task] = x_tasks[task] + self.drop_path1(
                    self.ls_1(attn_tasks_out[task])
                )

        # MLP block
        norm_x = self.ln_2(x)
        norm_x_tasks = (
            {task: self.ln_2(x_tasks[task]) for task in self.tasks} if x_tasks else None
        )

        # The MTLoRALinear forward needs to be called directly for the sequential MLP
        mlp_fc_out, mlp_fc_tasks_out = self.mlp.c_fc(norm_x, norm_x_tasks)
        gelu_out = self.mlp.gelu(mlp_fc_out)
        gelu_tasks_out = (
            {task: self.mlp.gelu(mlp_fc_tasks_out[task]) for task in self.tasks}
            if mlp_fc_tasks_out
            else None
        )

        mlp_proj_out, mlp_proj_tasks_out = self.mlp.c_proj(gelu_out, gelu_tasks_out)

        x = x + self.drop_path2(self.ls_2(mlp_proj_out))
        if mlp_proj_tasks_out and x_tasks:
            for task in self.tasks:
                x_tasks[task] = x_tasks[task] + self.drop_path2(
                    self.ls_2(mlp_proj_tasks_out[task])
                )

        return x, x_tasks

    def load_from_original_block(self, original_block):
        """
        Initializes the weights of this block from a pre-trained ResidualAttentionBlock.
        The LoRA-specific parameters are reset to their initial state.
        """
        with torch.no_grad():
            # Copy LayerNorm and LayerScale weights
            self.ln_1.load_state_dict(original_block.ln_1.state_dict())
            self.ln_2.load_state_dict(original_block.ln_2.state_dict())
            self.ls_1.load_state_dict(original_block.ls_1.state_dict())
            self.ls_2.load_state_dict(original_block.ls_2.state_dict())

            # Copy MLP weights into the .linear attribute of the MTLoRALinear layers
            self.mlp.c_fc.linear.load_state_dict(original_block.mlp.c_fc.state_dict())
            self.mlp.c_proj.linear.load_state_dict(
                original_block.mlp.c_proj.state_dict()
            )

            # Copy Attention weights
            # Both SelfAttention and nn.MultiheadAttention store QKV weights combined
            if isinstance(original_block.attn, SelfAttention):
                # Using migrate_weights ensures the Parameters are copied to the Linear layer first
                # Then we can extract from the Linear layer
                original_block.attn.migrate_weights()  # Ensure weights are in .in_proj and .out_proj

                # Split the combined weight and bias tensors into Q, K, V from .in_proj
                qkv_weight = original_block.attn.in_proj.weight
                qkv_bias = original_block.attn.in_proj.bias

                q_w, k_w, v_w = qkv_weight.chunk(3)
                q_b, k_b, v_b = qkv_bias.chunk(3)

                # Load into the .linear attributes of the MTLoRAQKV module
                self.attn.q.linear.weight.copy_(q_w)
                self.attn.q.linear.bias.copy_(q_b)

                self.attn.k.linear.weight.copy_(k_w)
                self.attn.k.linear.bias.copy_(k_b)

                self.attn.v.linear.weight.copy_(v_w)
                self.attn.v.linear.bias.copy_(v_b)

                # Load the output projection weights
                self.out_proj.linear.load_state_dict(
                    original_block.attn.out_proj.state_dict()
                )
            elif isinstance(original_block.attn, nn.MultiheadAttention):
                self.attn.q.linear.weight.copy_(
                    original_block.attn.in_proj_weight[
                        : self.attn.q.linear.out_features, :
                    ]
                )
                self.attn.q.linear.bias.copy_(
                    original_block.attn.in_proj_bias[: self.attn.q.linear.out_features]
                )

                self.attn.k.linear.weight.copy_(
                    original_block.attn.in_proj_weight[
                        self.attn.q.linear.out_features : 2
                        * self.attn.q.linear.out_features,
                        :,
                    ]
                )
                self.attn.k.linear.bias.copy_(
                    original_block.attn.in_proj_bias[
                        self.attn.q.linear.out_features : 2
                        * self.attn.q.linear.out_features
                    ]
                )

                self.attn.v.linear.weight.copy_(
                    original_block.attn.in_proj_weight[
                        2 * self.attn.q.linear.out_features : 3
                        * self.attn.q.linear.out_features,
                        :,
                    ]
                )
                self.attn.v.linear.bias.copy_(
                    original_block.attn.in_proj_bias[
                        2 * self.attn.q.linear.out_features : 3
                        * self.attn.q.linear.out_features
                    ]
                )

                self.out_proj.linear.weight.copy_(original_block.attn.out_proj.weight)
                self.out_proj.linear.bias.copy_(original_block.attn.out_proj.bias)

            else:
                raise TypeError(
                    f"Unsupported attention module type in original_block: {type(original_block.attn)}"
                )

        # After loading pretrained weights, re-initialize LoRA-specific parameters
        # This ensures that at the start of finetuning, the LoRA adjustment is zero.
        self.attn.reset_parameters()
        self.out_proj.reset_parameters()
        self.mlp.c_fc.reset_parameters()
        self.mlp.c_proj.reset_parameters()

        print(
            "Successfully loaded weights from original ResidualAttentionBlock and reset LoRA parameters."
        )


class MTLoRAAttentionPooling(nn.Module):
    """
    A  MT-LoRA equivalent of the AttentionPooling transformer block.

    This module replicates the full original architecture:
    1. Task-specific probes for attention pooling.
    2. MT-LoRA enabled Q/K/V and Output projections.
    3. A LayerNorm layer.
    4. An MLP block with MT-LoRA enabled linear layers.
    5. A final residual connection, matching the original's structure.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        tasks: List[str],
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        mlp_ratio: int = 4,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.tasks = tasks
        self.num_heads = num_heads

        self.probe = nn.ParameterDict(
            {task: nn.Parameter(torch.randn(1, 1, embed_dim)) for task in tasks}
        )

        task_scales = {t: lora_task_scale for t in tasks}

        self.q_proj = MTLoRALinear(
            embed_dim,
            embed_dim,
            r=r,
            lora_shared_scale=lora_shared_scale,
            lora_task_scale=task_scales,
            lora_dropout=lora_dropout,
            tasks=tasks,
        )
        self.k_proj = MTLoRALinear(
            embed_dim,
            embed_dim,
            r=r,
            lora_shared_scale=lora_shared_scale,
            lora_task_scale=task_scales,
            lora_dropout=lora_dropout,
            tasks=tasks,
        )
        self.v_proj = MTLoRALinear(
            embed_dim,
            embed_dim,
            r=r,
            lora_shared_scale=lora_shared_scale,
            lora_task_scale=task_scales,
            lora_dropout=lora_dropout,
            tasks=tasks,
        )
        self.out_proj = MTLoRALinear(
            embed_dim,
            embed_dim,
            r=r,
            lora_shared_scale=lora_shared_scale,
            lora_task_scale=task_scales,
            lora_dropout=lora_dropout,
            tasks=tasks,
        )

        self.layernorm = norm_layer(embed_dim)
        mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c_fc",
                        MTLoRALinear(
                            embed_dim,
                            mlp_width,
                            r=r,
                            lora_shared_scale=lora_shared_scale,
                            lora_task_scale=task_scales,
                            lora_dropout=lora_dropout,
                            tasks=tasks,
                        ),
                    ),
                    ("gelu", nn.GELU()),
                    (
                        "c_proj",
                        MTLoRALinear(
                            mlp_width,
                            embed_dim,
                            r=r,
                            lora_shared_scale=lora_shared_scale,
                            lora_task_scale=task_scales,
                            lora_dropout=lora_dropout,
                            tasks=tasks,
                        ),
                    ),
                ]
            )
        )

    def load_from_original(self, original_pool: AttentionPooling):
        """Initializes all weights from the pretrained AttentionPooling block."""
        with torch.no_grad():
            original_attn = original_pool.attn

            for task in self.tasks:
                self.probe[task].copy_(original_pool.probe)

            q_w, k_w, v_w = original_attn.in_proj_weight.chunk(3)
            q_b, k_b, v_b = original_attn.in_proj_bias.chunk(3)

            self.q_proj.linear.weight.copy_(q_w)
            self.q_proj.linear.bias.copy_(q_b)
            self.k_proj.linear.weight.copy_(k_w)
            self.k_proj.linear.bias.copy_(k_b)
            self.v_proj.linear.weight.copy_(v_w)
            self.v_proj.linear.bias.copy_(v_b)

            self.out_proj.linear.load_state_dict(original_attn.out_proj.state_dict())

            self.layernorm.load_state_dict(original_pool.layernorm.state_dict())

            self.mlp.c_fc.linear.load_state_dict(original_pool.mlp.c_fc.state_dict())
            self.mlp.c_proj.linear.load_state_dict(
                original_pool.mlp.c_proj.state_dict()
            )

        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.mlp.c_fc.reset_parameters()
        self.mlp.c_proj.reset_parameters()
        print(
            "Full MT-LoRA Attention Pooling block created and initialized from pretrained weights."
        )

    def forward(self, x_tasks: Dict[str, torch.Tensor]):
        """
        Forward pass that correctly handles unique inputs for each task.

        In this version, K and V are calculated inside the loop based on
        the task-specific input 'x', and the each task has it's unique probe.
        """

        final_outputs = {}
        for task, x in x_tasks.items():
            B, N, C = x.shape
            probe = self.probe[task].repeat(B, 1, 1)

            _, q_task_dict = self.q_proj(probe, x_tasks={task: probe})
            q = q_task_dict[task]

            _, k_task_dict = self.k_proj(x, x_tasks={task: x})
            k = k_task_dict[task]

            _, v_task_dict = self.v_proj(x, x_tasks={task: x})
            v = v_task_dict[task]

            q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out_rearranged = rearrange(attn_out, "b h n d -> b n (h d)")

            _, out_proj_dict = self.out_proj(
                attn_out_rearranged, x_tasks={task: attn_out_rearranged}
            )
            x_attn = out_proj_dict[task]

            norm_attn = self.layernorm(x_attn)

            _, fc_task_dict = self.mlp.c_fc(norm_attn, x_tasks={task: norm_attn})
            gelu_out = self.mlp.gelu(fc_task_dict[task])
            _, proj_task_dict = self.mlp.c_proj(gelu_out, x_tasks={task: gelu_out})
            mlp_out = proj_task_dict[task]

            final_outputs[task] = x_attn + mlp_out

        return final_outputs
