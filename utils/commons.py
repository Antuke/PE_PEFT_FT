"""Contains functions used for loading and logging models"""

import sys
import os
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

import os
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms_pe
from core.vision_encoder.config import PE_VISION_CONFIG
import torchvision.transforms as transforms
from PIL import Image
import requests


def print_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = (trainable_params / total_params * 100) if total_params > 0 else 0
    print("\n--- Summary ---")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Percentage:           {percent:.2f}%")


def get_backbone_pe(version, print_info=False, apply_migration_flag=False):
    """
    Load PE ViT model, return model, transforms and size of output (dimension of embedding of last token)
    """
    print(f"Loading {version}...")
    backbone = pe.VisionTransformer.from_config(version, pretrained=True)
    backbone_config = PE_VISION_CONFIG[version]
    transform = transforms_pe.get_image_transform_fix(
        image_size=backbone_config.image_size
    )

    print("\nYou can ignore the Missing keys list above.")
    print(f"Applying migration = {apply_migration_flag}")

    if print_info:
        attnpool = backbone.attn_pool
        print(f"embed_dim={attnpool.embed_dim}\nnum_heads={attnpool.num_heads}")
        print(f"OUTPUT DIM = {backbone_config.output_dim}")

    def apply_migration(m):
        if isinstance(m, pe.SelfAttention):
            m.migrate_weights()

    if (
        apply_migration_flag == True
    ):  # when testing/resuming no migration should be used
        print("[MIGRATION] Migrating weights for PEFT compatibiltyy")
        # Apply fn recursively to every submodule (as returned by .children()) as well as self.
        backbone.apply(apply_migration)

    return backbone, transform, backbone_config.output_dim


def get_backbone_dinov3(
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", print_info=False
):
    print(f"Loading Hugging Face model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    # Extract image processing configuration from the loaded processor
    image_processor_config = processor
    image_size = image_processor_config.size["height"]
    image_mean = image_processor_config.image_mean
    image_std = image_processor_config.image_std

    transform = transforms.Compose(
        [
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )

    # Load the model and return only the vision backbone
    vision_model = AutoModel.from_pretrained(model_name)

    if print_info:
        print(f"\nVISION CONFIGS:\n{vision_model.config}")
        print(f"\n\n\n{vision_model}")

    return vision_model, transform, vision_model.config.hidden_size


def get_backbone_siglip2(
    model_name: str = "google/siglip2-base-patch16-224", print_info=False
):
    """
    Load siglip2 ViT model, return model, transforms and size of output (dimension of embedding of last token)
    """
    print(f"Loading Hugging Face model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    # Extract image processing configuration from the loaded processor
    image_processor_config = processor.image_processor
    image_size = image_processor_config.size["height"]
    image_mean = image_processor_config.image_mean
    image_std = image_processor_config.image_std

    transform = transforms.Compose(
        [
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )

    # Load the model and return only the vision backbone
    model = AutoModel.from_pretrained(model_name)
    vision_model = model.vision_model

    if print_info:
        print(f"\nVISION CONFIGS:\n{vision_model.config}")
        print(f"\n\n***************MHAP\n{vision_model.head}")

    return vision_model, transform, vision_model.config.hidden_size


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """Converts a PIL Image to RGB format."""
    return image.convert("RGB")


def get_backbone(version: str, apply_migration: bool = False):
    """
    Returns vision transformer backbone
    Args:
        version: Name of the backbone to use, PE-Core or Siglip
        ckpt: if different from null, loads backbone from .pt file specified, only for PE
    """
    if "PE-Core-" in version:
        return get_backbone_pe(version, False, apply_migration)
    elif "siglip2" in version:
        print("[LOADING SIGLIP2]")
        return get_backbone_siglip2(version)
    elif "dinov3" in version:
        return get_backbone_dinov3(version)


def send_telegram_message(message: str):
    """Sends a message to a Telegram chat using credentials from the config."""
    # Get credentials from your config object. Use getattr for safety.
    token = os.getenv("BOT_TOKEN")
    chat_id = "1220514183"

    if not token or not chat_id:
        # Silently fail if credentials are not set
        return

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",  
    }

    try:
        response = requests.post(api_url, data=payload, timeout=10)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        # Don't crash the training loop if Telegram is down/token invalid
        print(f"\nWarning: Could not send Telegram message. Error: {e}")
