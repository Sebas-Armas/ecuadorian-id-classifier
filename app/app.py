"""
Document Classifier API — Ecuadorian Documents
Flask service that accepts an image and returns:
  { "tipo_documento": "cedula|pasaporte|desconocido", "confianza": 0.xx }
"""

import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, jsonify, render_template, request

# ── Path setup ────────────────────────────────────────────────────────────────
# Adjust these to point at your project's global/ folder and best_model.pth
BASE_DIR    = Path(__file__).resolve().parent.parent
print(BASE_DIR)
GLOBAL_PATH = str(BASE_DIR / "global")  # same as notebooks
print(GLOBAL_PATH)
if GLOBAL_PATH not in sys.path:
    sys.path.insert(0, GLOBAL_PATH)

import config  # noqa: E402  (project config)

# ── Constants ─────────────────────────────────────────────────────────────────
# Business-class mapping  (model has 4 classes; cedula = id_new + id_old)
MERGED_CLASS_NAMES = ["cedula", "pasaporte", "desconocido"]
SPEC_LABELS        = {0: "cedula", 1: "pasaporte", 2: "desconocido"}
LABEL_REMAP        = {0: 0, 1: 0, 2: 1, 3: 2}   # id_new/id_old→cedula, passport→pasaporte, unknown→desconocido

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_BYTES     = 10 * 1024 * 1024   # 10 MB hard limit

# ── Model loading (done once at startup) ─────────────────────────────────────
def load_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()
    print(f"[OK] Model loaded: {checkpoint['model_name']} | device: {config.DEVICE}")
    return model


# ── Inference transform (same as val/test in utils.py) ────────────────────────
def build_transform() -> A.Compose:
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


TRANSFORM = build_transform()


# ── Core prediction ───────────────────────────────────────────────────────────
def predict(model: torch.nn.Module, pil_image: Image.Image) -> dict:
    """
    Runs inference on a single PIL image.
    Returns the spec-compliant JSON dict.
    """
    # Convert to numpy RGB (albumentations expects numpy)
    img_np = np.array(pil_image.convert("RGB"))

    # Apply transforms and add batch dimension
    tensor = TRANSFORM(image=img_np)["image"].unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)                        # (1, 4)
        probs4 = F.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

    # Merge id_new + id_old → cedula
    probs3 = np.array([
        probs4[0] + probs4[1],  # cedula
        probs4[2],               # pasaporte
        probs4[3],               # desconocido
    ])

    idx        = int(np.argmax(probs3))
    confidence = float(probs3[idx])

    doc_type = (
        SPEC_LABELS[idx]
        if confidence >= config.CONFIDENCE_THRESHOLD
        else "desconocido"
    )

    return {
        "tipo_documento": doc_type,
        "confianza": round(confidence, 4),
        # Extra debug info (hidden from spec output, useful for UI)
        "_debug": {
            "cedula_prob":      round(float(probs3[0]), 4),
            "pasaporte_prob":   round(float(probs3[1]), 4),
            "desconocido_prob": round(float(probs3[2]), 4),
            "threshold":        config.CONFIDENCE_THRESHOLD,
        }
    }


# ── Flask app ──────────────────────────────────────────────────────────────────
app   = Flask(__name__)
MODEL = load_model(config.BEST_MODEL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # ── Validation ────────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image field in request. Send as multipart/form-data with key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: jpg, jpeg, png."}), 415

    raw = file.read()
    if len(raw) > MAX_FILE_BYTES:
        return jsonify({"error": f"File too large ({len(raw)//1024}KB). Max is 10MB."}), 413

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.verify()                    # catches corrupt files early
        pil_image = Image.open(io.BytesIO(raw))  # re-open after verify()
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {str(e)}"}), 422

    # ── Check minimum size ────────────────────────────────────────────────────
    w, h = pil_image.size
    if w < config.IMAGE_MIN_W or h < config.IMAGE_MIN_H:
        return jsonify({"error": f"Image too small ({w}x{h}). Minimum: {config.IMAGE_MIN_W}x{config.IMAGE_MIN_H}."}), 422

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = predict(MODEL, pil_image)
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # Return spec-compliant JSON (strip _debug for clean output)
    spec_output = {
        "tipo_documento": result["tipo_documento"],
        "confianza":      result["confianza"],
    }
    # Attach debug info only if requested (?debug=1)
    if request.args.get("debug") == "1":
        spec_output["debug"] = result["_debug"]

    return jsonify(spec_output), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(config.DEVICE)}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)