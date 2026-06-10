#!/usr/bin/env python3
from __future__ import annotations

"""
Browser-based Manuscript Annotation Tool - Multi-Class Adaptaton

Output per annotation:
    patches /           - {stem}_patch.png
    masks /             - {stem}_mask_binary.png (8-bit grayscale, 0/255)
                          {stem}_mask_rgba.png (RGBA, white ink / transparent bg)
                          {stem}_mask_values.png (single-channel 0/1)
    training_patches/ - train_{stem}.png (256 x 256 BGR image)
                        train_{stem}_mask.png (256 x 256 greyscale 0/255)
                        train_{stem}_mask_rgba.png (256x256 RGBA, a=255 where annotated)
    {image_stem}_session.json

stem = {image_stem}_ann_{ann_id}_{label}

Pipeline: _mask_rgba.png files feed directly into calvo_independent_train.py --background-mask / --layer-masks
"""

import argparse
import base64
import json
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=str(Path(__file__).parent))

# server side state
_current_image_path: Path | None = None
_output_dir: Path | None = None
_session_data: dict = {"image_path": "", "annotations": []}
_preload_image: str | None = None
_classes: list[str] = ["background", "staffline"]

# helpers
def _init_dirs(image_path: Path) -> Path:
    out = Path(__file__).parent.parent / "annotations"
    out.mkdir(exist_ok=True)
    (out / "patches").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    (out / "training_patches").mkdir(exist_ok=True)
    return out

def _export_training_patch(
        orig: np.ndarray,
        mask_binary: np.ndarray,
        bbox: dict,
        stem: str,
        out_dir: Path,
        patch_size: int = 256,
) -> tuple[Path, Path, Path]:
    """Returns (train_patch_path, train_mask_path, train_mask_rgba_path)"""
    img_h, img_w = orig.shape[:2]
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])

    cx, cy = x + w // 2, y + h // 2
    half = patch_size // 2
    px1 = max(0, cx - half)
    py1 = max(0, cy - half)
    px2 = min(img_w, cx + half)
    py2 = min(img_h, cy + half)

    train_patch = orig[py1:py2, px1:px2]

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[y : y + h, x : x + w] = mask_binary
    train_mask = full_mask[py1:py2, px1:px2]

    def _pad(arr: np.ndarray) -> np.ndarray:
        ph = patch_size - arr.shape[0]
        pw = patch_size - arr.shape[1]
        if ph > 0 or pw > 0:
            arr = np.pad(
                arr,
                ((0, ph), (0, pw) if arr.ndim == 2 else ((0, ph), (0, pw), (0, 0))),
                mode="constant",
                constant_values=0,
            )
        return arr
    
    train_patch = _pad(train_patch)
    train_mask = _pad(train_mask)

    # RGBA training mask: alpha=255 where annotated
    ps = train_mask.shape[0]
    train_mask_rgba = np.zeroes((ps, ps, 4), dtype=np.uint8)
    train_mask_rgba[train_mask > 0] = [255, 255, 255, 255]

    train_dir = out_dir / "training_patches"
    train_dir.mkdir(exist_ok=True)
    patch_p = train_dir / f"train_{stem}.png"
    mask_p = train_dir / f"train_{stem}_mask.png"
    mask_rgba_p = train_dir / f"train_{stem}_mask_rgba.png"
    cv2.imwrite(str(patch_p), train_patch)
    cv2.imwrite(str(mask_p), train_mask)

    return patch_p, mask_p, mask_rgba_p

def _load_session(image_path: Path, out_dir: Path) -> dict:
    sf = out_dir / f"{image_path.stem}_session.json"
    if sf.exists():
        with open(sf) as f:
            return json.load(f)
    return {"image_path": str(image_path), "annotations": []}

def _save_session(session: dict, image_path: Path, out_dir: Path):
    sf = out_dir / f"{image_path.stem}_session.json"
    with open(sf, "w") as f:
        json.dump(session, f, indent=2)

def _image_to_b64(path: Path) -> tuple[str, int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/png;base64,{b64}", img.shape[1], img.shape[0]

# routes
@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).resolve().parent), "independent_annotator.html")

@app.route("/api/preload")
def preload():
    return jsonify({"path": _preload_image})

@app.route("/api/classes")
def get_classes():
    return jsonify({"classes": _classes})

@app.route("/api/load_image", methods=["POST"])
def load_image():
    global _current_image_path, _output_dir, _session_data
    data = request.json
    path = Path(data["path"]).expanduser().resolve()
    if not path.exists():
        return jsonify({"error": f"File not found: {path}"}), 404
    
    _current_image_path = path
    _output_dir = _init_dirs(path)
    _session_data = _load_session(path, _output_dir)

    b64, w, h = _image_to_b64(path)
    return jsonify (
        {
            "image": b64,
            "width": w,
            "height": h,
            "name": path.stem,
            "annotations": _session_data.get("annotations", []),
        }
    )

@app.route("/api/upload_image", methods=["POST"])
def upload_image():
    global _current_image_path, _output_dir, _session_data
    file = request.files["image"]

    save_dir = Path(__file__).parent / "browser_uploads"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / file.filename
    file.save(str(save_path))

    _current_image_path = save_path
    _output_dir = _init_dirs(save_path)
    _session_data = _load_session(save_path, _output_dir)

    b64, w, h, = _image_to_b64(save_path)
    return jsonify(
        {
            "image": b64,
            "width": w,
            "height": h,
            "name": save_path.stem,
            "annotations": _session_data.get("annotations", []),
        }
    )

@app.route("/api/save_annotation", methods=["POST"])
def save_annotation():
    if _current_image_path is None:
        return jsonify({"error": "No image loaded"}), 400
    
    data = request.json
    bbox = data["bbox"]
    mask_b64 = data["mask"]
    label = data.get("label", _classes[0] if _classes else "background")
    ann_id = data.get("id", datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    orig = cv2.imread(str(_current_image_path))
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
    patch = orig[y : y + h, x : x + w]

    raw = base64.b64decode(mask_b64.split(",")[1] if "," in mask_b64 else mask_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return jsonify({"error": "Could not decode mask"}), 400
    if mask_img.shape[:2] != (h, w):
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    # label embedded in stem so every output filename is unambiguous per class
    stem = f"{_current_image_path.stem}_ann_{ann_id}_{label}"

    patch_p = _output_dir / "patches" / f"{stem}_patch.png"
    cv2.imwrite(str(patch_p), patch)

    bin_p = _output_dir / "masks" / f"{stem}_mask_binary.png"
    cv2.imwrite(str(bin_p), mask_binary)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_binary > 0] = [255, 255, 255, 255]
    rgba_p = _output_dir / "masks" / f"{stem}_mask_rgba.png"
    cv2.imwrite(str(rgba_p), rgba)

    val_mask = (mask_binary // 255).astype(np.uint8)
    val_p = _output_dir / "masks" / f"{stem}_mask_values.png"
    cv2.imwrite(str(val_p), val_mask)

    train_p, train_mask_p, train_mask_rgba_p = _export_training_patch(
        orig, mask_binary, bbox, stem, _output_dir
    )

    record = {
        "id": ann_id,
        "bbox": bbox,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "files": {
            "patch": str(patch_p),
            "mask_binary": str(bin_p),
            "mask_rgba": str(rgba_p),
            "mask_values": str(val_p),
            "train_patch": str(train_p),
            "train_mask": str(train_mask_p),
            "train_mask_rgba": str(train_mask_rgba_p) # direct input to calvo_independent_train
        },
    }
    existing = [a for a in _session_data["annotations"] if a["id"] != ann_id]
    existing.append(record)
    _session_data["annotations"] = existing
    _save_session(_session_data, _current_image_path, _output_dir)

    return jsonify({"success": True, "id": ann_id, "files": record["files"]})

@app.route("/api/delete_annotation", methods=["POST"])
def delete_annotation():
    if _current_image_path is None:
        return jsonify({"error": "No image loaded"}), 400
    ann_id = request.json["id"]

    for ann in _session_data["annotations"]:
        if ann["id"] == ann_id:
            for fpath in ann.get("files", {}).values():
                p = Path(fpath)
                if p.exists():
                    p.unlink()
            break
        
    _session_data["annotations"] = [
        a for a in _session_data["annotations"] if a["id"] != ann_id
    ]
    _save_session(_session_data, _current_image_path, _output_dir)
    return jsonify({"success": True})

# entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Class Browser Annotator")
    parser.add_argument("--image", type=str, help="Path to image (pre-loads on open)")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["background", "staffline"],
        help="Annotation class names.",
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    _preload_image = (
        str(Path(args.image).expanduser().resolve()) if args.image else None
    )
    _classes = args.classes

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"Annotator running at {url}")
    print(f"Classes: {_classes}")
    print(f"Press Ctrl-C to stop.\n")
    app.run(port=args.port, debug=False, use_reloader=False)
    