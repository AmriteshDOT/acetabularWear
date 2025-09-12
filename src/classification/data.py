import os
from pathlib import Path
import shutil
import tensorflow as tf
import random

img_size = (224, 224)
batch_size = 32
seed = 42
train_ratio = 0.8
src_folder = "all_patients"
out_folder = "patient_split"

patient_classes = {}


def organize_split(src, class_map, out_root, train_ratio=0.8, seed=42):
    src = Path(src)
    train_root = Path(out_root) / "train"
    val_root = Path(out_root) / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    patients = [p for p in os.listdir(src) if (src / p).is_dir()]
    patients.sort()
    random.seed(seed)
    random.shuffle(patients)

    split_idx = int(train_ratio * len(patients))
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]

    def copy_list(p_list, target):
        for p in p_list:
            cls = class_map.get(p)
            if not cls:
                continue
            src_p = src / p
            dst_p = target / cls / p
            dst_p.mkdir(parents=True, exist_ok=True)
            for img in src_p.glob("*.jpg"):
                shutil.copy2(img, dst_p / img.name)

    copy_list(train_patients, train_root)
    copy_list(val_patients, val_root)

    print(f"Train: {len(train_patients)} patients, Val: {len(val_patients)} patients")
    return train_root, val_root


def get_datasets(src, class_map, train_ratio=0.8):
    train_dir, val_dir = organize_split(src, class_map, out_folder, train_ratio, seed)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )

    classes = train_ds.class_names
    # print(f"classes: {classes}")

    auto = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(auto)
    val_ds = val_ds.cache().prefetch(auto)

    return train_ds, val_ds, classes


if __name__ == "__main__":
    train_ds, val_ds, classes = get_datasets(src_folder, patient_classes)
    # print(f"Train: {len(train_ds)}, Vals: {len(val_ds)}")
