"""
app.py — Flask microservice for Ecuadorian document classification.

This is the entry point for the inference API. It loads the trained model
once at startup, then serves predictions over HTTP so any other service
(PHP backend, .NET Core, a mobile app, etc.) can classify document images
without needing to bundle a Python ML stack itself.

The response always follows this contract:
    {"tipo_documento": "cedula | pasaporte | desconocido", "confianza": 0.xx}
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# The app lives in ecuadorian-id-classifier/app/, and the global/ folder
# and best_model.pth are two levels up (at the project root).
# We add global/ to sys.path so that "import config" just works.

BASE_DIR    = Path(__file__).resolve().parent.parent
GLOBAL_PATH = str(BASE_DIR / "global")
if GLOBAL_PATH not in sys.path:
    sys.path.insert(0, GLOBAL_PATH)

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------
# The model was trained on 4 classes (id_new, id_old, passport, unknown),
# but the API only exposes 3.  The two cédula variants are merged because
# from the caller's perspective it doesn't matter whether the document is
# the old green or the new blue format — it's still a cédula.
#
# LABEL_REMAP maps the raw model output index to one of the 3 public indices:
#   0 (id_new)  → 0 (cedula)
#   1 (id_old)  → 0 (cedula)
#   2 (passport)→ 1 (pasaporte)
#   3 (unknown) → 2 (desconocido)

SPEC_LABELS = {0: "cedula", 1: "pasaporte", 2: "desconocido"}
LABEL_REMAP = {0: 0, 1: 0, 2: 1, 3: 2}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_BYTES     = 10 * 1024 * 1024  # 10 MB — generous but not unlimited


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """
    Loads the trained checkpoint and reconstructs the model architecture.

    We saved the model name and number of classes inside the checkpoint dict
    (not just the weights), so we can always recreate the exact same
    architecture from the file alone — no need to hard-code it here.

    The model is set to eval() mode so dropout and batch-norm behave
    correctly during inference.
    """
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,           # we're loading our own weights, not ImageNet's
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    print(f"[OK] Model loaded: {checkpoint['model_name']} | device: {config.DEVICE} - {config.DEVICE_NAME}")
    return model


# ---------------------------------------------------------------------------
# Inference transform
# ---------------------------------------------------------------------------

def build_transform() -> A.Compose:
    """
    Builds the same preprocessing pipeline used during validation/testing.

    It's critical that this is identical to the val transform in utils.py.
    If the image preprocessing differs even slightly between training and
    inference (e.g. a different resize or missing normalisation), the model
    will see a distribution it was never trained on and predictions will be
    unreliable.
    """
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


TRANSFORM = build_transform()


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def predict(model: torch.nn.Module, pil_image: Image.Image) -> dict:
    """
    Runs a single image through the model and returns a prediction dict.

    Steps:
      1. Convert PIL → numpy RGB (albumentations expects numpy).
      2. Apply the inference transform and add a batch dimension.
      3. Run the forward pass with gradients disabled (faster, less memory).
      4. Convert raw logits to probabilities with softmax.
      5. Merge the two cédula classes into one probability.
      6. Apply the confidence threshold — below 0.70 we say "desconocido"
         rather than returning a low-confidence guess.

    The returned dict always has "tipo_documento" and "confianza".
    It also carries a "_debug" key with per-class probabilities that the
    /predict endpoint strips from normal responses but exposes on ?debug=1.
    """
    img_np = np.array(pil_image.convert("RGB"))
    tensor = TRANSFORM(image=img_np)["image"].unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)                               # shape: (1, 4)
        probs4 = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape: (4,)

    # Merge id_new + id_old → single "cedula" probability.
    # This is a simple sum because the two classes are mutually exclusive.
    probs3 = np.array([
        probs4[0] + probs4[1],  # cedula      (id_new + id_old)
        probs4[2],              # pasaporte
        probs4[3],              # desconocido
    ])

    idx        = int(np.argmax(probs3))
    confidence = float(probs3[idx])

    # Only commit to a label if the model is confident enough.
    # Below the threshold we fall back to "desconocido" so the caller
    # knows to treat the result with suspicion.
    doc_type = (
        SPEC_LABELS[idx]
        if confidence >= config.CONFIDENCE_THRESHOLD
        else "desconocido"
    )

    return {
        "tipo_documento": doc_type,
        "confianza": round(confidence, 4),
        "_debug": {
            "cedula_prob":      round(float(probs3[0]), 4),
            "pasaporte_prob":   round(float(probs3[1]), 4),
            "desconocido_prob": round(float(probs3[2]), 4),
            "threshold":        config.CONFIDENCE_THRESHOLD,
        },
    }


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
# The model is loaded here, at module level, so it happens exactly once
# when the server starts.  Loading it inside the route function would
# re-read the ~16 MB file on every request, which would be very slow.

app   = Flask(__name__)
MODEL = load_model(config.BEST_MODEL)


@app.route("/")
def index():
    """Serves the drag-and-drop web UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict

    Accepts a multipart/form-data request with a single field called "image"
    (jpg / jpeg / png, max 10 MB) and returns a JSON prediction.

    We validate the request thoroughly before touching the model because
    bad inputs are far more common than model errors in production, and
    it's better to return a clear 4xx than a cryptic 500.

    Add ?debug=1 to the URL to also receive per-class probabilities.
    """
    # Make sure the "image" field is actually present in the request.
    if "image" not in request.files:
        return jsonify({
            "error": "No image field in request. Send as multipart/form-data with key 'image'."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Allowed: jpg, jpeg, png."
        }), 415

    raw = file.read()
    if len(raw) > MAX_FILE_BYTES:
        return jsonify({
            "error": f"File too large ({len(raw) // 1024} KB). Max is 10 MB."
        }), 413

    # Try to open the image and catch corrupt files early.
    # Pillow's verify() consumes the stream, so we need to re-open from
    # the same byte buffer after calling it.
    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(raw))
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {str(e)}"}), 422

    # Reject images that are too small to contain a readable document.
    w, h = pil_image.size
    if w < config.IMAGE_MIN_W or h < config.IMAGE_MIN_H:
        return jsonify({
            "error": (
                f"Image too small ({w}x{h}). "
                f"Minimum: {config.IMAGE_MIN_W}x{config.IMAGE_MIN_H}."
            )
        }), 422

    # Run inference and wrap any unexpected error in a 500.
    try:
        result = predict(MODEL, pil_image)
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # Build the clean spec-compliant response.
    response = {
        "tipo_documento": result["tipo_documento"],
        "confianza":      result["confianza"],
    }

    # Only attach the debug breakdown if the caller explicitly asks for it.
    if request.args.get("debug") == "1":
        response["debug"] = result["_debug"]

    return jsonify(response), 200


@app.route("/health")
def health():
    """
    GET /health

    Quick liveness check.  Also reports the active compute device so you
    can confirm at a glance whether the GPU is being used.
    """
    return jsonify({"status": "ok", "device": str(config.DEVICE)}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)