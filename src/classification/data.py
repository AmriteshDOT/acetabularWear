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


def organize_split_3way(src, class_map, out_root, train_ratio=0.8, val_ratio=0.1, seed=42):

    src = Path(src)
    train_root = Path(out_root) / "train"
    val_root = Path(out_root) / "val"
    test_root = Path(out_root) / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    patients = [p for p in os.listdir(src) if (src / p).is_dir()]
    patients.sort()
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    def copy_list(p_list, target_root):
        for p in p_list:
            cls = class_map.get(p)
            if not cls:
                continue
            src_p = src / p
            dst_p = target_root / cls / p
            dst_p.mkdir(parents=True, exist_ok=True)
            for img in src_p.glob("*.jpg"):
                shutil.copy2(img, dst_p / img.name)

    copy_list(train_patients, train_root)
    copy_list(val_patients, val_root)
    copy_list(test_patients, test_root)

    # print(f"Train: {len(train_patients)} patients, Val: {len(val_patients)} patients, Test: {len(test_patients)} patients")
    return train_root, val_root, test_root


def get_datasets(src, class_map, train_ratio=0.8, val_ratio=0.1):
    train_dir, val_dir, test_dir = organize_split_3way(src, class_map, out_folder, train_ratio, val_ratio, seed)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, seed=seed
    )

    classes = train_ds.class_names
    # print(f"classes: {classes}")

    auto = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(auto)
    val_ds = val_ds.cache().prefetch(auto)
    test_ds = test_ds.cache().prefetch(auto)

    return train_ds, val_ds, test_ds, classes


if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes = get_datasets(src_folder, patient_classes)
    # print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")
