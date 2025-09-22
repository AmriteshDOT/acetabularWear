# metrics_key.py
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import numpy as np


def plot_history(hist, out=None):
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    acc = hist.history.get("accuracy", hist.history.get("acc", []))
    val_acc = hist.history.get("val_accuracy", hist.history.get("val_acc", []))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    if out:
        plt.savefig(out, bbox_inches="tight")
    plt.show()
    plt.close()


def eval_metrics(model, ds, classes, out=None):
    ys = []
    ypred = []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        ypred.extend(p.argmax(axis=1))
        ys.extend(yb.numpy())
    ys = np.array(ys)
    ypred = np.array(ypred)

    acc = accuracy_score(ys, ypred)
    prec, rec, f1, sup = precision_recall_fscore_support(ys, ypred, average=None)
    cm = confusion_matrix(ys, ypred)


    for i, c in enumerate(classes):
        print(
            f"{c} -> precision: {prec[i]}, recall: {rec[i]}, f1: {f1[i]}, support: {sup[i]}"
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Pred")
    plt.ylabel("True")
    if out:
        plt.savefig(out, bbox_inches="tight")
    plt.show()
    plt.close()

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}

