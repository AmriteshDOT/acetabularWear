# patient_split_minimal_clean.py
from pathlib import Path
import shutil
import random


def organize_split_3way(
    src,
    class_map,
    out_root,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    exts=("jpg", "jpeg", "png"),
):
    src = Path(src)
    tr = Path(out_root) / "train"
    va = Path(out_root) / "val"
    te = Path(out_root) / "test"
    tr.mkdir(parents=True, exist_ok=True)
    va.mkdir(parents=True, exist_ok=True)
    te.mkdir(parents=True, exist_ok=True)

    pats = [p.name for p in src.iterdir() if p.is_dir()]
    pats.sort()
    rnd = random.Random(seed)
    rnd.shuffle(pats)

    n = len(pats)
    ntr = int(train_ratio * n)
    nva = int(val_ratio * n)
    tr_p = pats[:ntr]
    va_p = pats[ntr : ntr + nva]
    te_p = pats[ntr + nva :]

    def copy_list(plist, dst):
        for p in plist:
            cls = class_map[p]
            s = src / p
            d = dst / cls / p
            d.mkdir(parents=True, exist_ok=True)
            for ext in exts:
                for f in s.glob(f"*.{ext}"):
                    shutil.copy2(f, d / f.name)

    copy_list(tr_p, tr)
    copy_list(va_p, va)
    copy_list(te_p, te)

    return tr, va, te


def get_datasets(
    train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32, seed=42
):
    import tensorflow as tf

    tr_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )
    va_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )
    te_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )

    classes = tr_ds.class_names
    A = tf.data.AUTOTUNE
    tr_ds = tr_ds.cache().shuffle(1000).prefetch(A)
    va_ds = va_ds.cache().prefetch(A)
    te_ds = te_ds.cache().prefetch(A)

    return tr_ds, va_ds, te_ds, classes


def main():
    src = "all_patients"
    out = "patient_split"
    cls_map = {}  # {"p1": "normal", .}
    tr, va, te = organize_split_3way(src, cls_map, out)
    train_ds, val_ds, test_ds, classes = get_datasets(tr, va, te)
    print("done. classes:", classes)


if __name__ == "__main__":
    main()
