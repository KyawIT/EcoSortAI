#!/usr/bin/env python3
"""
Convert ecosort.keras → ecosort.onnx.

Bypasses tf.saved_model.save entirely to avoid the seed_generator/untracked-
resource error that RandomFlip/Rotation augmentation layers cause in Keras 3.
Uses a traced @tf.function (training=False) with tf2onnx.from_function —
no SavedModel intermediate needed.

Usage:
    pip install tf2onnx
    python scripts/convert_to_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = ROOT / "model" / "ecosort.keras"
ONNX_MODEL = ROOT / "model" / "ecosort.onnx"


def main() -> None:
    if not KERAS_MODEL.exists():
        print(f"[ERROR] Model not found: {KERAS_MODEL}")
        sys.exit(1)

    print("Loading model …")
    import tensorflow as tf  # noqa: PLC0415
    import tf2onnx  # noqa: PLC0415

    model = tf.keras.models.load_model(str(KERAS_MODEL))

    h = model.input_shape[1] or 224
    w = model.input_shape[2] or 224
    print(f"Input shape: ({h}, {w}, 3)")

    # Trace with training=False so augmentation layers become identity ops
    spec = (tf.TensorSpec(shape=[None, h, w, 3], dtype=tf.float32, name="image"),)

    @tf.function(input_signature=spec)
    def inference(x):
        return model(x, training=False)

    print("Tracing inference function …")
    # Warm-up trace to make debugging easier; conversion uses the tf.function
    # object, not the returned ConcreteFunction.
    inference.get_concrete_function()

    print("Converting to ONNX (opset 17) …")
    model_proto, _ = tf2onnx.convert.from_function(
        inference,
        input_signature=spec,
        opset=17,
    )

    ONNX_MODEL.write_bytes(model_proto.SerializeToString())

    size_mb = ONNX_MODEL.stat().st_size / 1_048_576
    print(f"[OK] {ONNX_MODEL} ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  git add model/ecosort.onnx")
    print("  git commit -m 'add ONNX model'")
    print("  git push")


if __name__ == "__main__":
    main()
