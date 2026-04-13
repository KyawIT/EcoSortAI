#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
class ModelInputSpec:
    name: str
    shape: Sequence[Any]
    dtype: str


@dataclass
class ModelArtifacts:
    session: ort.InferenceSession
    image_input_name: str
    channel_first: bool
    inputs: List[ModelInputSpec]
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


def resolve_hwc_layout(shape: Sequence[Any]) -> Tuple[int, int, bool]:
    """Infer image layout from ONNX input shape.

    Returns (height, width, channel_first).
    Defaults to NHWC if uncertain.
    """
    if len(shape) != 4:
        return 224, 224, False

    # Typical NHWC: [N, H, W, 3]
    if isinstance(shape[3], int) and shape[3] == 3:
        h = shape[1] if isinstance(shape[1], int) else 224
        w = shape[2] if isinstance(shape[2], int) else 224
        return h, w, False

    # Typical NCHW: [N, 3, H, W]
    if isinstance(shape[1], int) and shape[1] == 3:
        h = shape[2] if isinstance(shape[2], int) else 224
        w = shape[3] if isinstance(shape[3], int) else 224
        return h, w, True

    h = shape[1] if isinstance(shape[1], int) else 224
    w = shape[2] if isinstance(shape[2], int) else 224
    return h, w, False


def ort_dtype_to_numpy(ort_type: str) -> np.dtype:
    mapping: Dict[str, np.dtype] = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(ort_type, np.float32)


def materialize_shape(shape: Sequence[Any], batch_size: int) -> Tuple[int, ...]:
    dims: List[int] = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        elif idx == 0:
            dims.append(batch_size)
        else:
            dims.append(1)
    return tuple(dims)


def build_aux_input(spec: ModelInputSpec, batch_size: int) -> np.ndarray:
    """Create fallback values for non-image ONNX inputs.

    Some tf2onnx exports expose normalization tensors as additional required
    graph inputs. We provide stable defaults to satisfy these signatures.
    """
    np_dtype = ort_dtype_to_numpy(spec.dtype)
    shape = materialize_shape(spec.shape, batch_size)
    value = np.zeros(shape, dtype=np_dtype)
    name = spec.name.lower()

    if shape and shape[-1] == 3 and ("sub/y" in name or "mean" in name):
        rgb = np.asarray([0.485, 0.456, 0.406], dtype=np_dtype)
        value[...] = rgb.reshape((1,) * (len(shape) - 1) + (3,))
    elif shape and shape[-1] == 3 and ("sqrt/x" in name or "std" in name):
        rgb = np.asarray([0.229, 0.224, 0.225], dtype=np_dtype)
        value[...] = rgb.reshape((1,) * (len(shape) - 1) + (3,))
    elif "sqrt" in name or "std" in name or "variance" in name:
        value[...] = np.asarray(1.0, dtype=np_dtype)

    return value


def load_artifacts(model_path: Path, labels_path: Path) -> ModelArtifacts:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    class_names = read_class_names(labels_path)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_infos = session.get_inputs()
    if not input_infos:
        raise ValueError("ONNX model has no inputs.")

    image_input = next((x for x in input_infos if x.name == "image"), None)
    if image_input is None:
        image_input = next((x for x in input_infos if len(x.shape) == 4), input_infos[0])

    height, width, channel_first = resolve_hwc_layout(image_input.shape)
    inputs = [ModelInputSpec(name=x.name, shape=x.shape, dtype=x.type) for x in input_infos]

    return ModelArtifacts(
        session=session,
        image_input_name=image_input.name,
        channel_first=channel_first,
        inputs=inputs,
        class_names=class_names,
        image_height=height,
        image_width=width,
        model_path=str(model_path),
    )


def preprocess_image(image_bytes: bytes, height: int, width: int, channel_first: bool) -> np.ndarray:
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
    array = np.expand_dims(array, axis=0)  # (1, H, W, 3)
    if channel_first:
        array = np.transpose(array, (0, 3, 1, 2))
    return array


def predict_top_k(artifacts: ModelArtifacts, image_bytes: bytes, top_k: int) -> Dict[str, Any]:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    image_array = preprocess_image(
        image_bytes=image_bytes,
        height=artifacts.image_height,
        width=artifacts.image_width,
        channel_first=artifacts.channel_first,
    )

    input_feed: Dict[str, np.ndarray] = {}
    batch_size = int(image_array.shape[0]) if image_array.ndim > 0 else 1
    for spec in artifacts.inputs:
        if spec.name == artifacts.image_input_name:
            input_feed[spec.name] = image_array
        else:
            input_feed[spec.name] = build_aux_input(spec, batch_size=batch_size)

    outputs = artifacts.session.run(None, input_feed)
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
        "model_inputs": [
            {"name": i.name, "shape": list(i.shape), "dtype": i.dtype} for i in artifacts.inputs
        ],
        "image_input_name": artifacts.image_input_name,
        "channel_first": artifacts.channel_first,
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
