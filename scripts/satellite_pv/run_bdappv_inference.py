from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = REPO_ROOT / "satellite_experiment" / "full_shanghai_dataset"
DEFAULT_INPUT = FULL_ROOT / "data" / "shanghai_imagery_index.csv"
DEFAULT_OUTPUT = FULL_ROOT / "results" / "bdappv_predictions.csv"
DEFAULT_MODEL_DIR = FULL_ROOT / "models" / "bdappv"
DEFAULT_MASK_DIR = FULL_ROOT / "masks"
DEFAULT_OVERLAY_DIR = FULL_ROOT / "overlays"
HF_REPO = "gabrielkasmi/bdappv-models"
MODEL_HELPER_URL = f"https://huggingface.co/{HF_REPO}/raw/main/model.py"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def ensure_model_helper(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    helper_path = model_dir / "model.py"
    if not helper_path.exists():
        response = requests.get(MODEL_HELPER_URL, timeout=60)
        response.raise_for_status()
        helper_path.write_text(response.text, encoding="utf-8")
    return helper_path


def load_bdappv_helper(model_dir: Path):
    helper_path = ensure_model_helper(model_dir)
    spec = importlib.util.spec_from_file_location("bdappv_model", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_windows(width: int, height: int, window_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    if width <= window_size or height <= window_size:
        return [(0, 0, width, height)]

    xs = list(range(0, max(1, width - window_size + 1), stride))
    ys = list(range(0, max(1, height - window_size + 1), stride))
    if xs[-1] != width - window_size:
        xs.append(width - window_size)
    if ys[-1] != height - window_size:
        ys.append(height - window_size)
    return [(x, y, x + window_size, y + window_size) for y in ys for x in xs]


def classify_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    window_size: int,
    stride: int,
) -> tuple[float, tuple[int, int, int, int], list[float], list[tuple[int, int, int, int]]]:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    windows = make_windows(image.width, image.height, window_size, stride)
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            tensors = []
            for box in batch_windows:
                crop = image.crop(box).convert("RGB")
                tensors.append(transform(crop))
            batch = torch.stack(tensors).to(device)
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits.flatten()).detach().cpu().numpy()
            scores.extend(float(value) for value in probs)

    best_index = int(np.argmax(scores)) if scores else 0
    best_score = scores[best_index] if scores else math.nan
    best_window = windows[best_index] if windows else (0, 0, image.width, image.height)
    return best_score, best_window, scores, windows


def segment_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: str,
    windows: list[tuple[int, int, int, int]],
    scores: list[float],
    score_threshold: float,
    batch_size: int,
) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    selected = [(window, score) for window, score in zip(windows, scores) if score >= score_threshold]
    if not selected and windows:
        best_index = int(np.argmax(scores))
        selected = [(windows[best_index], scores[best_index])]

    full_mask = np.zeros((image.height, image.width), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_items = selected[start : start + batch_size]
            tensors = []
            for box, _score in batch_items:
                crop = image.crop(box).convert("RGB")
                tensors.append(transform(crop))
            batch = torch.stack(tensors).to(device)
            output = model(batch)
            logits = output["out"] if isinstance(output, dict) else output
            probs = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
            for (box, _score), prob in zip(batch_items, probs):
                x1, y1, x2, y2 = box
                mask_img = Image.fromarray((prob * 255).astype("uint8")).resize(
                    (x2 - x1, y2 - y1), Image.Resampling.BILINEAR
                )
                mask = np.asarray(mask_img, dtype=np.float32) / 255.0
                full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)
    return full_mask


def save_mask_and_overlay(
    image: Image.Image,
    mask: np.ndarray,
    image_stem: str,
    mask_dir: Path,
    overlay_dir: Path,
    threshold: float,
) -> tuple[str, str, float]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    binary = mask >= threshold
    mask_ratio = float(binary.mean())
    mask_path = mask_dir / f"{image_stem}_mask.png"
    overlay_path = overlay_dir / f"{image_stem}_overlay.jpg"

    Image.fromarray((binary.astype("uint8") * 255)).save(mask_path)

    base = image.convert("RGBA")
    color = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    color[binary] = [255, 64, 32, 120]
    overlay = Image.alpha_composite(base, Image.fromarray(color, mode="RGBA")).convert("RGB")
    overlay.save(overlay_path, quality=94)
    return str(mask_path), str(overlay_path), mask_ratio


def status_from_score(score: float, possible_threshold: float, likely_threshold: float) -> str:
    if score >= likely_threshold:
        return "likely_pv"
    if score >= possible_threshold:
        return "possible_pv"
    return "no_clear_pv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BDAPPV open-source PV recognition.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--provider", choices=["google", "ign"], default="google")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--stride", type=int, default=312)
    parser.add_argument("--possible-threshold", type=float, default=0.45)
    parser.add_argument("--likely-threshold", type=float, default=0.75)
    parser.add_argument("--segmentation-threshold", type=float, default=0.50)
    parser.add_argument("--segment-score-threshold", type=float, default=0.45)
    parser.add_argument("--run-segmentation", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    fields, rows = read_csv(args.input)
    rows = rows[: args.limit] if args.limit is not None else rows

    module = load_bdappv_helper(args.model_dir)
    classifier = module.load_classification_model(args.provider, device=args.device)
    segmenter = (
        module.load_segmentation_model(args.provider, device=args.device)
        if args.run_segmentation
        else None
    )

    result_fields = fields + [
        "bdappv_provider",
        "pv_score_max",
        "pv_status_model",
        "best_window",
        "window_scores",
        "mask_path",
        "overlay_path",
        "mask_ratio",
        "inference_error",
    ]
    results: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        out_row = dict(row)
        out_row.update(
            {
                "bdappv_provider": args.provider,
                "pv_score_max": "",
                "pv_status_model": "error",
                "best_window": "",
                "window_scores": "",
                "mask_path": "",
                "overlay_path": "",
                "mask_ratio": "",
                "inference_error": "",
            }
        )

        try:
            image_path = Path(row["image_path"])
            image = Image.open(image_path).convert("RGB")
            score, best_window, scores, windows = classify_image(
                image=image,
                model=classifier,
                device=args.device,
                batch_size=args.batch_size,
                window_size=args.window_size,
                stride=args.stride,
            )
            out_row["pv_score_max"] = f"{score:.6f}"
            out_row["pv_status_model"] = status_from_score(
                score, args.possible_threshold, args.likely_threshold
            )
            out_row["best_window"] = ",".join(str(value) for value in best_window)
            out_row["window_scores"] = ";".join(f"{value:.4f}" for value in scores)

            if segmenter is not None and score >= args.segment_score_threshold:
                mask = segment_image(
                    image=image,
                    model=segmenter,
                    device=args.device,
                    windows=windows,
                    scores=scores,
                    score_threshold=args.segment_score_threshold,
                    batch_size=max(1, min(args.batch_size, 4)),
                )
                mask_path, overlay_path, mask_ratio = save_mask_and_overlay(
                    image=image,
                    mask=mask,
                    image_stem=image_path.stem,
                    mask_dir=args.mask_dir,
                    overlay_dir=args.overlay_dir,
                    threshold=args.segmentation_threshold,
                )
                out_row["mask_path"] = mask_path
                out_row["overlay_path"] = overlay_path
                out_row["mask_ratio"] = f"{mask_ratio:.8f}"
        except Exception as exc:
            out_row["inference_error"] = repr(exc)

        results.append(out_row)
        if index % args.checkpoint_every == 0:
            write_csv(args.output, result_fields, results)
            print(f"inference_checkpoint={index}/{len(rows)}")

    write_csv(args.output, result_fields, results)
    status_counts: dict[str, int] = {}
    for row in results:
        status = row["pv_status_model"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"prediction_output={args.output}")
    print(f"prediction_rows={len(results)}")
    print(f"status_counts={status_counts}")


if __name__ == "__main__":
    main()
