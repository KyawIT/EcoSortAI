#!/usr/bin/env python3
"""
Train EcoSort image classifier with transfer learning (EfficientNetB0).

Expected dataset layout:
data/split/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Keep Matplotlib cache writable inside the project sandbox.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EcoSort classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        help="Pretrained weights source: 'imagenet', 'none', or path to local .h5 file.",
    )
    parser.add_argument(
        "--unfreeze-percent",
        type=float,
        default=0.2,
        help="Fraction of top base-model layers to unfreeze in fine-tuning (0.0-1.0).",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--reduce-lr-patience", type=int, default=2)
    parser.add_argument("--no-confusion-plot", action="store_true")
    return parser.parse_args()


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.is_file())


def load_datasets(
    data_dir: Path, image_size: int, batch_size: int, seed: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for required in (train_dir, val_dir, test_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing directory: {required}. "
                "Run split step first to generate train/val/test folders."
            )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    class_names = list(train_ds.class_names)

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def build_model(
    num_classes: int,
    image_size: int,
    dropout_rate: float,
    weights: str,
) -> Tuple[keras.Model, keras.Model]:
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    if weights.lower() == "none":
        base_weights = None
    else:
        base_weights = weights

    try:
        base_model = EfficientNetB0(
            include_top=False,
            weights=base_weights,
            input_shape=(image_size, image_size, 3),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize EfficientNetB0 weights. "
            "If you are offline, run with '--weights none'."
        ) from exc
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ecosort_efficientnetb0")
    return model, base_model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def make_callbacks(
    checkpoint_path: Path,
    early_stopping_patience: int,
    reduce_lr_patience: int,
) -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def history_to_csv(
    histories: List[Tuple[str, keras.callbacks.History]], output_csv: Path
) -> None:
    rows = []
    for phase, history in histories:
        if history is None:
            continue
        metrics = history.history
        n_epochs = len(metrics.get("loss", []))
        for i in range(n_epochs):
            rows.append(
                {
                    "phase": phase,
                    "epoch": i + 1,
                    "loss": metrics.get("loss", [None] * n_epochs)[i],
                    "accuracy": metrics.get("accuracy", [None] * n_epochs)[i],
                    "val_loss": metrics.get("val_loss", [None] * n_epochs)[i],
                    "val_accuracy": metrics.get("val_accuracy", [None] * n_epochs)[i],
                }
            )

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def save_confusion_plot(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_artifacts(
    output_dir: Path,
    class_names: List[str],
    test_metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    no_confusion_plot: bool,
) -> None:
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    labels_payload = {
        "index_to_label": class_names,
        "label_to_index": label_to_index,
    }
    (output_dir / "labels.json").write_text(json.dumps(labels_payload, indent=2))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    if not no_confusion_plot:
        save_confusion_plot(cm, class_names, output_dir / "confusion_matrix.png")

    (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ecosort_best.keras"
    final_model_path = output_dir / "ecosort.keras"

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(
        "Images: "
        f"train={count_images(data_dir / 'train')}, "
        f"val={count_images(data_dir / 'val')}, "
        f"test={count_images(data_dir / 'test')}"
    )

    model, base_model = build_model(
        num_classes=len(class_names),
        image_size=args.image_size,
        dropout_rate=args.dropout_rate,
        weights=args.weights,
    )

    histories: List[Tuple[str, keras.callbacks.History]] = []

    print("\nStage 1/2: Train classification head")
    compile_model(model, args.head_learning_rate)
    if args.head_epochs > 0:
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs,
            callbacks=make_callbacks(
                checkpoint_path=checkpoint_path,
                early_stopping_patience=args.early_stopping_patience,
                reduce_lr_patience=args.reduce_lr_patience,
            ),
            verbose=1,
        )
        histories.append(("head", head_history))
    else:
        print("Skipped (head-epochs=0)")
        head_history = None

    if args.fine_tune_epochs > 0:
        print("\nStage 2/2: Fine-tune top EfficientNet layers")
        base_model.trainable = True
        total_layers = len(base_model.layers)
        unfreeze_percent = min(max(args.unfreeze_percent, 0.0), 1.0)
        fine_tune_at = int(total_layers * (1.0 - unfreeze_percent))
        fine_tune_at = min(max(fine_tune_at, 0), total_layers)

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print(
            f"Fine-tune config: unfreezing top {total_layers - fine_tune_at} "
            f"of {total_layers} base layers"
        )

        compile_model(model, args.fine_tune_learning_rate)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=make_callbacks(
                checkpoint_path=checkpoint_path,
                early_stopping_patience=args.early_stopping_patience,
                reduce_lr_patience=args.reduce_lr_patience,
            ),
            verbose=1,
        )
        histories.append(("fine_tune", fine_history))
    else:
        print("\nStage 2/2: Skipped (fine-tune-epochs=0)")

    history_to_csv(histories, output_dir / "history.csv")

    if checkpoint_path.exists():
        best_model = keras.models.load_model(checkpoint_path)
    else:
        best_model = model

    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    y_prob = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)

    test_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num_classes": len(class_names),
        "classes": class_names,
    }

    save_artifacts(
        output_dir=output_dir,
        class_names=class_names,
        test_metrics=test_metrics,
        y_true=y_true,
        y_pred=y_pred,
        no_confusion_plot=args.no_confusion_plot,
    )

    best_model.save(final_model_path)
    print("\nTraining complete.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Final model saved: {final_model_path}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
