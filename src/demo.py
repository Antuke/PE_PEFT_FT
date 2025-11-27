import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont  #
from tqdm import tqdm

from utils.commons import get_backbone
from src.model import MTLModel
from utils.task_config import Task
from utils.face_detector import FaceDetector
from src.utils.config_parser import parse_config


def get_adaptive_font_scale(
    text, img_width, img_height, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1.5
):
    """
    Calculate adaptive font scale based on image dimensions.
    """
    # Start with a base scale
    font_scale = 0.7

    # Get text size with base scale
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Text should not exceed 90% of image width
    width_scale = (img_width * 0.9) / text_width if text_width > 0 else 1.0
    # Text height should not exceed 20% of image height for better vertical fit
    height_scale = (img_height * 0.2) / text_height if text_height > 0 else 1.0

    # Use the smaller scale to ensure text fits
    adaptive_scale = min(width_scale, height_scale, 1.0) * font_scale

    # Ensure minimum readability
    return max(adaptive_scale, 0.3)


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Task Learning Inference Script")

    # Paths and environment
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./demo_images",
        help="Directory of images to classify",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./predicted",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the model checkpoint to load",
    )

    # Detection configuration
    parser.add_argument(
        "--detection_confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection",
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
        "--use_peft",
        action="store_true",
        help="Use PEFT LoRA architecture in the model",
    )
    parser.add_argument(
        "--use_mtlora",
        action="store_true",
        help="Use MT-LoRA architecture in the model",
    )
    parser.add_argument(
        "--use_deep_head",
        action="store_true",
        help="Use a prediction head with one hidden layer",
    )
    parser.add_argument(
        "--load_heads",
        action="store_true",
        help="Use a prediction head with one hidden layer",
    )
    parser.add_argument(
        "--only_most_confident",
        action="store_true",
        help="Detect only one face per image",
    )
    # Tasks configuration
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Age", "Gender", "Emotion"],
        choices=["Age", "Gender", "Emotion"],
        help="Specify which tasks to evaluate.",
    )

    args = parse_config(parser)
    if not args.ckpt_path:
        parser.error("the following arguments are required: --ckpt_path")
    return args


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Output Folder Setup ---
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.join(args.input_dir, "output")
        print(f"Output folder not specified. Using default: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    output_crop_folder = os.path.join(output_folder, "crop")
    output_bbox_folder = os.path.join(output_folder, "annotated")
    output_text_folder = os.path.join(output_folder, "annotated", "labels")
    os.makedirs(output_crop_folder, exist_ok=True)
    os.makedirs(output_bbox_folder, exist_ok=True)
    os.makedirs(output_text_folder, exist_ok=True)

    # --- Model and Transform Setup ---
    backbone, transform, _ = get_backbone(args.backbone_name, apply_migration=True)
    res_block_layers = len(backbone.transformer.resblocks)

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
        raise ValueError("No valid tasks selected.")

    model = MTLModel(
        backbone,
        tasks=TASKS,
        rank=args.rank,
        truncate_idx=(res_block_layers - 2),
        use_lora=args.use_peft,
        use_mtl_lora=args.use_mtlora,
        use_deep_head=args.use_deep_head,
    ).to(device)

    if args.use_peft or args.use_mtlora:
        model.load_adapters_peft(args.ckpt_path)
        model.backbone.merge_and_unload()
    else:
        model.load_trained_legacy(args.ckpt_path)

    print(f"Model loaded successfully from: {args.ckpt_path}")
    model.eval()

    # --- Face Detector ---
    detector = FaceDetector(confidence_threshold=args.detection_confidence)

    # --- Image Processing ---
    images = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"\nFound {len(images)} images in {args.input_dir}")

    for img_path in tqdm(images, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        faces = detector.detect(img, pad_rect=True)

        if faces is None:
            continue

        if args.only_most_confident:
            faces = [max(faces, key=lambda f: f[1])]

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        txt_lines = []

        for idx, (crop, confidence, bbox) in enumerate(faces):
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil_face = Image.fromarray(crop_rgb)
            crop_tensor = transform(crop_pil_face).unsqueeze(0).to(device)
            with torch.no_grad():
                result = model(crop_tensor)
            predictions = {}
            for task in TASKS:
                task_logits = result[task.name][0]
                probabilities = F.softmax(task_logits, dim=0)
                top2_conf, top2_idx = torch.topk(probabilities, 2)
                predictions[task.name] = {
                    "top1": (
                        task.class_labels[top2_idx[0].item()],
                        top2_conf[0].item(),
                    ),
                    "top2": (
                        task.class_labels[top2_idx[1].item()],
                        top2_conf[1].item(),
                    ),
                }
            pred_str_top1 = ", ".join(
                [
                    predictions.get(t.name, {"top1": ("N/A", 0)})["top1"][0]
                    for t in TASKS
                ]
            )
            crop_annotated = crop.copy()
            crop_h, crop_w = crop_annotated.shape[:2]
            crop_font_scale = get_adaptive_font_scale(
                pred_str_top1, crop_w, crop_h, thickness=1
            )
            cv2.putText(
                crop_annotated,
                pred_str_top1,
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                crop_font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            crop_path = os.path.join(output_crop_folder, f"{base_name}_face{idx}.jpg")
            cv2.imwrite(crop_path, crop_annotated)

            x, y, w, h = bbox

            font_size_ratio = 0.08
            min_font_size = 12
            max_font_size = 48
            adaptive_font_size = max(
                min_font_size, min(int(w * font_size_ratio), max_font_size)
            )
            try:
                font = ImageFont.load_default(size=adaptive_font_size)
            except IOError:
                font = ImageFont.load_default()

            draw.rectangle([(x, y), (x + w, y + h)], outline="lime", width=2)

            lines_to_draw = []
            for task in TASKS:
                top1_label, top1_conf = predictions[task.name]["top1"]
                top2_label, top2_conf = predictions[task.name]["top2"]
                line = f"{task.name[0]}: {top1_label} ({top1_conf * 100:.0f}%), {top2_label} ({top2_conf * 100:.0f}%)"
                lines_to_draw.append(line)

            # --- Calculate total height of the text block to decide placement ---
            line_spacing = 5
            total_text_height = 0
            for line in lines_to_draw:
                # Use textbbox to get the size of the text line
                _left, top, _right, bottom = draw.textbbox((0, 0), line, font=font)
                total_text_height += (bottom - top) + line_spacing

            # --- Place text ABOVE or BELOW the box ---
            if y - total_text_height > 0:
                # PLACE TEXT ABOVE: There is enough space
                text_y = y - line_spacing
                for line in reversed(lines_to_draw):
                    left, top, right, bottom = draw.textbbox(
                        (x, text_y), line, font=font, anchor="ls"
                    )  # anchor left-baseline
                    draw.rectangle(
                        [(left - 2, top - 2), (right + 2, bottom + 2)], fill="black"
                    )
                    draw.text((x, text_y), line, font=font, fill="white", anchor="ls")
                    text_y = top - line_spacing  # Move y-position up for the next line
            else:
                # PLACE TEXT BELOW: Not enough space above, so draw downwards
                text_y = y + h + line_spacing
                for line in lines_to_draw:
                    left, top, right, bottom = draw.textbbox(
                        (x, text_y), line, font=font, anchor="lt"
                    )
                    draw.rectangle(
                        [(left - 2, top - 2), (right + 2, bottom + 2)], fill="black"
                    )
                    draw.text((x, text_y), line, font=font, fill="white", anchor="lt")
                    text_y = bottom + line_spacing

            # --- Prepare text file line ---
            line_parts = []
            for task_name in ["Age", "Gender", "Emotion"]:
                if task_name in predictions:
                    top1_label, top1_prob = predictions[task_name]["top1"]
                    top2_label, top2_prob = predictions[task_name]["top2"]
                    line_parts.extend(
                        [
                            f"{top1_label}",
                            f"{top1_prob:.4f}",
                            f"{top2_label}",
                            f"{top2_prob:.4f}",
                        ]
                    )
                else:
                    line_parts.extend(["-", "0.0000", "-", "0.0000"])
            bbox_str = f"{x},{y},{w},{h}"
            line_parts.append(bbox_str)
            txt_lines.append(";".join(line_parts))

            if args.only_most_confident:
                break

        # --- Save the final annotated image ---
        img_bbox_annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        bbox_path = os.path.join(output_bbox_folder, f"{base_name}_annotated.png")
        cv2.imwrite(bbox_path, img_bbox_annotated)

        txt_path = os.path.join(output_text_folder, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(txt_lines))


if __name__ == "__main__":
    main()
