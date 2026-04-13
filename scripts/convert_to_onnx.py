#!/usr/bin/env python3
"""
Convert ecosort.keras → ecosort.onnx via SavedModel.

tf2onnx --keras flag doesn't support Keras 3 (set_learning_phase removed).
Workaround: save as SavedModel first, then convert that.

Usage:
    pip install tf2onnx
    python scripts/convert_to_onnx.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = ROOT / "model" / "ecosort.keras"
SAVED_MODEL = ROOT / "model" / "_ecosort_saved_model_tmp"
ONNX_MODEL = ROOT / "model" / "ecosort.onnx"


def main() -> None:
    if not KERAS_MODEL.exists():
        print(f"[ERROR] Model not found: {KERAS_MODEL}")
        sys.exit(1)

    # Step 1 — load .keras and re-export as SavedModel
    print("Step 1/2 — loading .keras and exporting as SavedModel ...")
    import tensorflow as tf  # noqa: PLC0415

    model = tf.keras.models.load_model(str(KERAS_MODEL))
    if SAVED_MODEL.exists():
        shutil.rmtree(SAVED_MODEL)
    tf.saved_model.save(model, str(SAVED_MODEL))
    print(f"          SavedModel written to {SAVED_MODEL}")

    # Step 2 — convert SavedModel → ONNX
    print("Step 2/2 — converting SavedModel → ONNX ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", str(SAVED_MODEL),
            "--output", str(ONNX_MODEL),
            "--opset", "17",
        ],
        check=False,
    )

    # Clean up temp SavedModel regardless of outcome
    shutil.rmtree(SAVED_MODEL, ignore_errors=True)

    if result.returncode != 0:
        print("[ERROR] ONNX conversion failed.")
        sys.exit(1)

    size_mb = ONNX_MODEL.stat().st_size / 1_048_576
    print(f"[OK] {ONNX_MODEL} ({size_mb:.1f} MB)")
    print("Next steps:")
    print("  git add model/ecosort.onnx")
    print("  git commit -m 'add ONNX model'")
    print("  git push")


if __name__ == "__main__":
    main()
