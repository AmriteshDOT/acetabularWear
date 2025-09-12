import os
import shutil
import random


def patient_wise_split(source_dir, output_dir, train_ratio=0.8, seed=42):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    patients = [
        p for p in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, p))
    ]
    random.seed(seed)
    random.shuffle(patients)

    split_idx = int(train_ratio * len(patients))
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]

    def copy_patients(patients, target_dir):
        for patient in patients:
            src = os.path.join(source_dir, patient)
            dst = os.path.join(target_dir, patient)
            if os.path.exists(dst):
                shutil.rmtree(dst)  # overwrite
            shutil.copytree(src, dst)

    copy_patients(train_patients, train_dir)
    copy_patients(val_patients, val_dir)

    # print(f"Train: {len(train_patients)}, Val: {len(val_patients)}")
    return train_dir, val_dir
