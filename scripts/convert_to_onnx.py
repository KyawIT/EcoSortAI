#!/usr/bin/env python3
"""
Convert ecosort.keras → ecosort.onnx for CPU deployment without AVX2 requirement.

Usage:
    pip install tf2onnx
    python scripts/convert_to_onnx.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = ROOT / "model" / "ecosort.keras"
ONNX_MODEL = ROOT / "model" / "ecosort.onnx"


def main() -> None:
    if not KERAS_MODEL.exists():
        print(f"[ERROR] Model not found: {KERAS_MODEL}")
        sys.exit(1)

    print(f"Converting {KERAS_MODEL.name} → {ONNX_MODEL.name} ...")

    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--keras", str(KERAS_MODEL),
            "--output", str(ONNX_MODEL),
            "--opset", "17",
        ],
        check=False,
    )

    if result.returncode != 0:
        print("[ERROR] Conversion failed. Make sure tf2onnx is installed: pip install tf2onnx")
        sys.exit(1)

    size_mb = ONNX_MODEL.stat().st_size / 1_048_576
    print(f"[OK] {ONNX_MODEL} ({size_mb:.1f} MB)")
    print("Next: git add model/ecosort.onnx && git commit -m 'add ONNX model'")


if __name__ == "__main__":
    main()
