import os
import shutil
import random


def patient_wise_split_3way(
    source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42
):

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    patients = [
        p for p in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, p))
    ]
    patients.sort()
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    def copy_patients(p_list, target_root):
        for patient in p_list:
            src = os.path.join(source_dir, patient)
            dst = os.path.join(target_root, patient)
            if os.path.exists(dst):
                shutil.rmtree(dst)  # overwrite , clean
            shutil.copytree(src, dst)

    copy_patients(train_patients, train_dir)
    copy_patients(val_patients, val_dir)
    copy_patients(test_patients, test_dir)

    return train_dir, val_dir, test_dir
