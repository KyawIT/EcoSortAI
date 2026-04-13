#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "ecosort.onnx"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "model" / "labels.json"

MODEL_PATH = Path(os.getenv("ECOSORT_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
LABELS_PATH = Path(os.getenv("ECOSORT_LABELS_PATH", str(DEFAULT_LABELS_PATH))).resolve()

BIN_GUIDE: Dict[str, Dict[str, str]] = {
    "battery": {
        "bin": "Battery / E-Waste Collection",
        "tip": "Do not dispose in regular trash or recycling bins.",
    },
    "glass": {
        "bin": "Glass Recycling Bin",
        "tip": "Rinse containers before recycling.",
    },
    "metal": {
        "bin": "Metal Recycling Bin",
        "tip": "Clean food residue when possible.",
    },
    "organic": {
        "bin": "Compost / Organic Waste Bin",
        "tip": "Use compostable liners if available.",
    },
    "paper": {
        "bin": "Paper Recycling Bin",
        "tip": "Keep paper dry and free from food stains.",
    },
    "plastic": {
        "bin": "Plastic Recycling Bin",
        "tip": "Rinse and flatten containers when possible.",
    },
}


@dataclass
class ModelArtifacts:
    session: ort.InferenceSession
    input_name: str
    class_names: List[str]
    image_height: int
    image_width: int
    model_path: str


def parse_allowed_origins() -> List[str]:
    raw = os.getenv("ECOSORT_ALLOWED_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def read_class_names(labels_path: Path) -> List[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")
    payload = json.loads(labels_path.read_text())
    class_names = payload.get("index_to_label")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("labels.json missing non-empty 'index_to_label' list")
    return [str(n) for n in class_names]


def load_artifacts(model_path: Path, labels_path: Path) -> ModelArtifacts:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    class_names = read_class_names(labels_path)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    shape = input_info.shape  # e.g. [None, 224, 224, 3]
    height = shape[1] if isinstance(shape[1], int) else 224
    width = shape[2] if isinstance(shape[2], int) else 224

    return ModelArtifacts(
        session=session,
        input_name=input_info.name,
        class_names=class_names,
        image_height=height,
        image_width=width,
        model_path=str(model_path),
    )


def preprocess_image(image_bytes: bytes, height: int, width: int) -> np.ndarray:
    """Resize and convert to float32 array.

    Preprocessing (EfficientNet torch-mode normalisation) is baked into the
    exported ONNX graph as a Lambda layer, so we feed raw [0, 255] pixels.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    image = image.resize((width, height))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)  # (1, H, W, 3)


def predict_top_k(artifacts: ModelArtifacts, image_bytes: bytes, top_k: int) -> Dict[str, Any]:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    image_array = preprocess_image(image_bytes, artifacts.image_height, artifacts.image_width)

    outputs = artifacts.session.run(None, {artifacts.input_name: image_array})
    probabilities: np.ndarray = outputs[0][0]  # (num_classes,)

    max_k = min(top_k, len(artifacts.class_names))
    top_indices = np.argsort(probabilities)[::-1][:max_k]

    top_predictions = []
    for idx in top_indices:
        class_name = artifacts.class_names[int(idx)]
        guide = BIN_GUIDE.get(class_name, {"bin": "Unknown", "tip": "No guidance available."})
        top_predictions.append({
            "class": class_name,
            "confidence": round(float(probabilities[int(idx)]), 6),
            "recommended_bin": guide["bin"],
            "tip": guide["tip"],
        })

    best = top_predictions[0]
    return {
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "recommended_bin": best["recommended_bin"],
        "tip": best["tip"],
        "top_predictions": top_predictions,
    }


app = FastAPI(title="EcoSort AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    app.state.artifacts = load_artifacts(model_path=MODEL_PATH, labels_path=LABELS_PATH)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "EcoSort AI backend is running."}


@app.get("/health")
def health() -> Dict[str, Any]:
    artifacts: ModelArtifacts | None = getattr(app.state, "artifacts", None)
    if artifacts is None:
        return {"status": "error", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": artifacts.model_path,
        "input_size": {"height": artifacts.image_height, "width": artifacts.image_width},
        "classes": artifacts.class_names,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=6),
) -> Dict[str, Any]:
    artifacts: ModelArtifacts | None = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return predict_top_k(artifacts=artifacts, image_bytes=image_bytes, top_k=top_k)
