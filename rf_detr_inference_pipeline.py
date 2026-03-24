#!/usr/bin/env python3
from __future__ import annotations

"""
Stable RF-DETR inference pipeline for fine-tuned checkpoints.

Supports:
- checkpoint inspection
- automatic model-class inference
- single image inference
- folder inference
- video inference
- annotated outputs
- JSON predictions
- robust fallback handling
- class-count alignment
- optional inference optimization

Install:
    pip install rfdetr supervision opencv-python pillow torch torchvision numpy tqdm

Examples:
    python rf_detr_inference_pipeline.py --checkpoint rf_detr_model/weights.pt --inspect-only

    python rf_detr_inference_pipeline.py \
        --checkpoint rf_detr_model/weights.pt \
        --input runs/test/images.png \
        --output-dir runs/infer \
        --force-model-class RFDETRMedium

    python rf_detr_inference_pipeline.py \
        --checkpoint rf_detr_model/weights.pt \
        --input path/to/images_folder \
        --output-dir runs/infer_batch

    python rf_detr_inference_pipeline.py \
        --checkpoint rf_detr_model/weights.pt \
        --input path/to/video.mp4 \
        --output-dir runs/infer_video
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Checkpoint inspection helpers
# -----------------------------

@dataclass
class CheckpointSummary:
    checkpoint_path: str
    checkpoint_format: str
    inferred_model_size: str
    inferred_model_class: str
    inferred_resolution: Optional[int]
    num_classes: Optional[int]
    class_names: List[str]
    class_head_outputs: Optional[int]
    total_parameters: Optional[int]
    has_args: bool
    has_model_key: bool
    notes: List[str]


def _safe_torch_load(path: str) -> Any:
    import argparse as _argparse

    try:
        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:
        pass

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_checkpoint(
    loaded: Any,
) -> Tuple[Dict[str, torch.Tensor], Optional[Any], str, bool, bool]:
    if isinstance(loaded, dict) and "model" in loaded:
        model_state = loaded["model"]
        args_obj = loaded.get("args")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint 'model' entry is not a state_dict.")
        return model_state, args_obj, "wrapped_state_dict", args_obj is not None, True

    if isinstance(loaded, dict) and loaded and all(torch.is_tensor(v) for v in loaded.values()):
        return loaded, None, "state_dict", False, False

    raise ValueError(
        "Unsupported checkpoint layout. Expected either a wrapped checkpoint "
        "with {'model': ..., 'args': ...} or a raw PyTorch state_dict."
    )


def _infer_model_size(
    args_obj: Optional[Any],
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[str, str, Optional[int], List[str]]:
    notes: List[str] = []
    encoder = getattr(args_obj, "encoder", None)
    resolution = getattr(args_obj, "resolution", None)
    total_params = sum(v.numel() for v in state_dict.values())

    if encoder == "dinov2_windowed_small" and resolution == 384:
        return "nano", "RFDETRNano", resolution, notes
    if encoder == "dinov2_windowed_small" and resolution == 512:
        return "small", "RFDETRSmall", resolution, notes
    if encoder == "dinov2_windowed_small" and resolution == 576:
        return "medium", "RFDETRMedium", resolution, notes
    if resolution == 704:
        return "large", "RFDETRLarge", resolution, notes
    if resolution == 700:
        return "xlarge", "RFDETRXLarge", resolution, notes
    if resolution == 880:
        return "2xlarge", "RFDETR2XLarge", resolution, notes

    if 32_000_000 <= total_params <= 35_500_000:
        notes.append("Model size inferred heuristically from parameter count.")
        return "medium", "RFDETRMedium", resolution, notes

    notes.append("Could not confidently infer model size from metadata. Defaulting to RFDETRMedium.")
    return "medium", "RFDETRMedium", resolution, notes


def inspect_checkpoint(checkpoint_path: str) -> CheckpointSummary:
    loaded = _safe_torch_load(checkpoint_path)
    state_dict, args_obj, checkpoint_format, has_args, has_model_key = _normalize_checkpoint(loaded)
    size_name, model_class, resolution, notes = _infer_model_size(args_obj, state_dict)

    class_names = list(getattr(args_obj, "class_names", []) or [])
    num_classes = getattr(args_obj, "num_classes", None)
    class_head_outputs = None

    if "class_embed.weight" in state_dict:
        class_head_outputs = int(state_dict["class_embed.weight"].shape[0])

    total_parameters = sum(v.numel() for v in state_dict.values())

    if num_classes is not None and class_names and len(class_names) != num_classes:
        notes.append(
            f"Metadata mismatch: num_classes={num_classes} but class_names has {len(class_names)} entries."
        )

    if class_head_outputs is not None and num_classes is not None:
        if class_head_outputs not in (num_classes, num_classes + 1):
            notes.append(
                f"Class head output count ({class_head_outputs}) does not match expected "
                f"num_classes ({num_classes}) or num_classes+1."
            )

    return CheckpointSummary(
        checkpoint_path=str(checkpoint_path),
        checkpoint_format=checkpoint_format,
        inferred_model_size=size_name,
        inferred_model_class=model_class,
        inferred_resolution=resolution,
        num_classes=num_classes,
        class_names=class_names,
        class_head_outputs=class_head_outputs,
        total_parameters=total_parameters,
        has_args=has_args,
        has_model_key=has_model_key,
        notes=notes,
    )


# -----------------------------
# RF-DETR loading helpers
# -----------------------------

def _import_model_class(model_class_name: str):
    try:
        from rfdetr import (
            RFDETR2XLarge,
            RFDETRLarge,
            RFDETRMedium,
            RFDETRNano,
            RFDETRSmall,
            RFDETRXLarge,
        )
    except ImportError as exc:
        raise ImportError("The 'rfdetr' package is not installed. Run: pip install rfdetr") from exc

    model_map = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
        "RFDETRXLarge": RFDETRXLarge,
        "RFDETR2XLarge": RFDETR2XLarge,
    }

    if model_class_name not in model_map:
        raise ValueError(f"Unsupported RF-DETR model class: {model_class_name}")

    return model_map[model_class_name]


def _extract_state_dict_for_fallback(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    loaded = _safe_torch_load(checkpoint_path)
    state_dict, _, _, _, _ = _normalize_checkpoint(loaded)
    return state_dict


def _sanitize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned[new_key] = value
    return cleaned


def build_model(checkpoint_path: str, force_model_class: Optional[str] = None):
    summary = inspect_checkpoint(checkpoint_path)
    model_class_name = force_model_class or summary.inferred_model_class
    model_cls = _import_model_class(model_class_name)

    load_errors: List[str] = []
    model_kwargs: Dict[str, Any] = {}

    if summary.num_classes is not None:
        model_kwargs["num_classes"] = summary.num_classes

    try:
        model = model_cls(pretrain_weights=checkpoint_path, **model_kwargs)
    except Exception as exc:
        load_errors.append(f"Direct checkpoint load failed: {exc}")

        try:
            model = model_cls(**model_kwargs)
            state_dict = _extract_state_dict_for_fallback(checkpoint_path)
            state_dict = _sanitize_state_dict_keys(state_dict)

            target = getattr(model, "model", model)
            incompatible = target.load_state_dict(state_dict, strict=False)

            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))

            if missing:
                summary.notes.append(f"Fallback load missing keys: {len(missing)}")
            if unexpected:
                summary.notes.append(f"Fallback load unexpected keys: {len(unexpected)}")
            if load_errors:
                summary.notes.extend(load_errors)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to load checkpoint with both direct and fallback strategies. "
                f"Previous errors: {load_errors}. Fallback error: {fallback_exc}"
            ) from fallback_exc

    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
            summary.notes.append("Model optimized for inference.")
        except Exception as exc:
            summary.notes.append(f"optimize_for_inference() failed: {exc}")

    return model, summary


# -----------------------------
# Prediction normalization
# -----------------------------

def _coerce_detections(predictions: Any):
    try:
        import supervision as sv
    except ImportError as exc:
        raise ImportError("The 'supervision' package is not installed. Run: pip install supervision") from exc

    if isinstance(predictions, sv.Detections):
        return predictions

    if isinstance(predictions, dict):
        try:
            return sv.Detections.from_inference(predictions)
        except Exception:
            pass

        try:
            if {"xyxy", "confidence", "class_id"}.issubset(predictions.keys()):
                return sv.Detections(
                    xyxy=np.asarray(predictions["xyxy"]),
                    confidence=np.asarray(predictions["confidence"]),
                    class_id=np.asarray(predictions["class_id"]),
                )
        except Exception:
            pass

    if hasattr(predictions, "xyxy") and hasattr(predictions, "confidence") and hasattr(predictions, "class_id"):
        return sv.Detections(
            xyxy=np.asarray(predictions.xyxy),
            confidence=np.asarray(predictions.confidence),
            class_id=np.asarray(predictions.class_id),
        )

    if isinstance(predictions, (list, tuple)) and len(predictions) == 3:
        try:
            return sv.Detections(
                xyxy=np.asarray(predictions[0]),
                confidence=np.asarray(predictions[1]),
                class_id=np.asarray(predictions[2]),
            )
        except Exception:
            pass

    raise TypeError(
        f"Could not normalize model output to supervision.Detections. Output type: {type(predictions)}"
    )


# -----------------------------
# Label and JSON helpers
# -----------------------------

def _normalize_class_names(class_names: Sequence[str], summary: CheckpointSummary) -> List[str]:
    names = list(class_names or [])

    expected = summary.num_classes
    if expected is None:
        return names

    if len(names) < expected:
        names = names + [f"class_{i}" for i in range(len(names), expected)]
    elif len(names) > expected:
        names = names[:expected]

    return names


def _resolve_labels(detections, class_names: Sequence[str]) -> List[str]:
    labels: List[str] = []
    class_id = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)

    if class_id is None:
        return labels

    class_id = np.asarray(class_id)
    confidence = np.asarray(confidence) if confidence is not None else None

    for i, cid in enumerate(class_id.tolist()):
        cid = int(cid)
        name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        if confidence is not None and i < len(confidence):
            labels.append(f"{name} {float(confidence[i]):.2f}")
        else:
            labels.append(name)

    return labels


def detections_to_json_ready(detections, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))))
    confidence = np.asarray(getattr(detections, "confidence", np.empty((0,))))
    class_id = np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=int)))

    items: List[Dict[str, Any]] = []
    for i in range(len(xyxy)):
        cid = int(class_id[i]) if i < len(class_id) else -1
        items.append(
            {
                "class_id": cid,
                "class_name": class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}",
                "confidence": float(confidence[i]) if i < len(confidence) else None,
                "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
            }
        )
    return items


# -----------------------------
# Annotation helpers
# -----------------------------

def annotate_rgb_image(rgb: np.ndarray, detections, class_names: Sequence[str]) -> np.ndarray:
    import supervision as sv

    labels = _resolve_labels(detections, class_names)
    image = rgb.copy()

    try:
        image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
    except TypeError:
        image = sv.BoxAnnotator().annotate(image, detections)

    try:
        image = sv.LabelAnnotator().annotate(scene=image, detections=detections, labels=labels)
    except TypeError:
        image = sv.LabelAnnotator().annotate(image, detections, labels)

    return image


# -----------------------------
# Inference runners
# -----------------------------

def predict_image(model, image_path: Path, threshold: float):
    image = Image.open(image_path).convert("RGB")
    raw = model.predict(image, threshold=threshold)
    detections = _coerce_detections(raw)
    return image, detections


def save_image_outputs(
    image_path: Path,
    pil_image: Image.Image,
    detections,
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    rgb = np.array(pil_image)
    annotated = annotate_rgb_image(rgb, detections, class_names)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{stem}_annotated.jpg"), annotated_bgr)

    pred_json = {
        "source": str(image_path),
        "predictions": detections_to_json_ready(detections, class_names),
    }
    (output_dir / f"{stem}_predictions.json").write_text(
        json.dumps(pred_json, indent=2),
        encoding="utf-8",
    )


def iter_images(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def run_on_image(model, image_path: Path, class_names: Sequence[str], threshold: float, output_dir: Path) -> None:
    pil_image, detections = predict_image(model, image_path, threshold)
    save_image_outputs(image_path, pil_image, detections, class_names, output_dir)


def run_on_folder(model, folder: Path, class_names: Sequence[str], threshold: float, output_dir: Path) -> None:
    images = list(iter_images(folder))
    if not images:
        raise FileNotFoundError(f"No images found under: {folder}")

    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    batch_json: List[Dict[str, Any]] = []

    for image_path in tqdm(images, desc="Processing images"):
        pil_image, detections = predict_image(model, image_path, threshold)
        save_image_outputs(image_path, pil_image, detections, class_names, image_output_dir)
        batch_json.append(
            {
                "source": str(image_path),
                "predictions": detections_to_json_ready(detections, class_names),
            }
        )

    (output_dir / "batch_predictions.json").write_text(
        json.dumps(batch_json, indent=2),
        encoding="utf-8",
    )


def run_on_video(model, video_path: Path, class_names: Sequence[str], threshold: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_predictions: List[Dict[str, Any]] = []
    frame_idx = 0
    progress = None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            raw = model.predict(pil, threshold=threshold)
            detections = _coerce_detections(raw)

            annotated_rgb = annotate_rgb_image(frame_rgb, detections, class_names)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

            frame_predictions.append(
                {
                    "frame_index": frame_idx,
                    "predictions": detections_to_json_ready(detections, class_names),
                }
            )
            frame_idx += 1
            progress.update(1)
    finally:
        if progress is not None:
            try:
                progress.close()
            except Exception:
                pass
        cap.release()
        writer.release()

    (output_dir / f"{video_path.stem}_predictions.json").write_text(
        json.dumps(frame_predictions, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable RF-DETR inference pipeline")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint (.pt / .pth)")
    parser.add_argument("--input", help="Path to an image, folder of images, or video")
    parser.add_argument("--output-dir", default="runs/rfdetr_infer", help="Directory for outputs")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument(
        "--force-model-class",
        choices=[
            "RFDETRNano",
            "RFDETRSmall",
            "RFDETRMedium",
            "RFDETRLarge",
            "RFDETRXLarge",
            "RFDETR2XLarge",
        ],
        help="Override automatic model-class inference",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect the checkpoint and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary = inspect_checkpoint(str(checkpoint_path))
    except Exception as exc:
        print(f"Failed to inspect checkpoint: {exc}", file=sys.stderr)
        return 1

    (output_dir / "checkpoint_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(summary), indent=2))

    if args.inspect_only:
        return 0

    if not args.input:
        print("--input is required unless --inspect-only is used.", file=sys.stderr)
        return 2

    try:
        model, summary = build_model(str(checkpoint_path), force_model_class=args.force_model_class)
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    class_names = summary.class_names
    if not class_names:
        model_class_names = getattr(model, "class_names", None)
        if isinstance(model_class_names, (list, tuple)):
            class_names = list(model_class_names)
    class_names = _normalize_class_names(class_names, summary)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        return 2

    suffix = input_path.suffix.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

    try:
        if input_path.is_file() and suffix in image_exts:
            run_on_image(model, input_path, class_names, args.threshold, output_dir)
        elif input_path.is_dir():
            run_on_folder(model, input_path, class_names, args.threshold, output_dir)
        elif input_path.is_file() and suffix in video_exts:
            run_on_video(model, input_path, class_names, args.threshold, output_dir)
        else:
            print(
                "Unsupported input. Provide an image file, a directory of images, or a video file.",
                file=sys.stderr,
            )
            return 2
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())