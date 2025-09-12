import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

img_size = (224, 224)
batch_size = 32


def load_test_ds(test_dir, img_size=img_size, batch_size=batch_size):
    test_dir = str(Path(test_dir))
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds


def simple_eval(model_path, test_dir, out_dir="eval_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)

    test_ds = load_test_ds(test_dir)

    loss, acc = model.evaluate(test_ds, verbose=1)
    summary = {"loss": float(loss), "accuracy": float(acc)}

    y_true = []
    y_pred = []

    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    classes = test_ds.class_names

    rep_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )
    df = pd.DataFrame(rep_dict).transpose()
    report_csv = out / "classification_report.csv"
    df.to_csv(report_csv, index=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("confusion matrix")
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_png = out / "confusion_matrix.png"
    fig.savefig(cm_png, dpi=150)
    plt.close(fig)

    summary["confusion_matrix"] = str(cm_png)
    summary["report_csv"] = str(report_csv)

    summary_json = out / "eval_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
