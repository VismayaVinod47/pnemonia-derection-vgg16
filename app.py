from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple

from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms

# Replace this import with your model architecture.
# Example: from model_definition import FederatedVGG
from model_definition import FederatedVGG

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "federated_vgg.pt"

app = Flask(__name__)


# IMPORTANT: Keep this transform aligned with your training transform.
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Set your class names in training label order.
CLASS_NAMES = ["Normal", "Pneumonia"]
NORMAL_CLASS_INDEX = 0
INFECTED_CLASS_INDEX = 1


def load_trained_model() -> torch.nn.Module:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    out_features = state_dict["classifier.6.weight"].shape[0]
    model = FederatedVGG(num_classes=out_features)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model: torch.nn.Module, img: Image.Image) -> Tuple[str, float, int]:
    if img.mode != "RGB":
        img = img.convert("RGB")

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        if logits.shape[1] == 1:
            prob_pos = torch.sigmoid(logits)[0, 0].item()
            pred_index = 1 if prob_pos >= 0.5 else 0
            confidence = (prob_pos if pred_index == 1 else 1.0 - prob_pos) * 100.0
        else:
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_index = pred_idx.item()
            confidence = conf.item() * 100.0

    label = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"class_{pred_index}"
    return label, confidence, pred_index


def get_result_theme(pred_index: int) -> str:
    if pred_index == NORMAL_CLASS_INDEX:
        return "normal"
    if pred_index == INFECTED_CLASS_INDEX:
        return "infected"
    return "neutral"


def validate_upload(mime_type: str, img: Image.Image) -> Tuple[bool, str]:
    if not mime_type.startswith("image/"):
        return False, "Invalid input. Please upload an image file."

    rgb_img = img.convert("RGB").resize((224, 224))
    tensor = transforms.ToTensor()(rgb_img)
    r, g, b = tensor[0], tensor[1], tensor[2]

    # Chest X-rays are usually grayscale-like, so channel differences
    # should stay relatively low.
    color_gap = (torch.abs(r - g).mean() + torch.abs(g - b).mean() + torch.abs(r - b).mean()) / 3
    if color_gap.item() > 0.09:
        return False, "Invalid input. Please upload a chest X-ray image."

    return True, ""


model = load_trained_model()


@app.route("/", methods=["GET"])
def landing():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    result_theme = None
    image_preview = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                raw_bytes = file.read()
                mime_type = file.mimetype or "image/jpeg"
                image_preview = f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"
                image = Image.open(BytesIO(raw_bytes))
                is_valid, validation_error = validate_upload(mime_type, image)
                if not is_valid:
                    error = validation_error
                else:
                    prediction, confidence, pred_index = predict_image(model, image)
                    result_theme = get_result_theme(pred_index)
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        result_theme=result_theme,
        image_preview=image_preview,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
