import shutil
import random
from pathlib import Path
import yaml
import glob


def prepare_by_patient(src_root, dst_root, train_ratio=0.7, val_ratio=0.15, seed=42):
    src = Path(src_root)
    dst = Path(dst_root)

    # folders
    subfolders = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]
    for s in subfolders:
        (dst / s).mkdir(parents=True, exist_ok=True)

    # patient folders
    patients = []
    for d in sorted(src.iterdir()):
        if d.is_dir():
            patients.append(d.name)

    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    # img & label
    def copy_patient_files(p_list, split):
        for pid in p_list:
            pdir = src / pid
            for ext in ("jpg", "jpeg", "png"):
                for img in sorted(pdir.glob(f"*.{ext}")):
                    fn = img.name
                    lbl = img.stem + ".txt"
                    shutil.copy2(str(img), dst / f"images/{split}" / fn)
                    lbl_file = pdir / lbl
                    if lbl_file.exists():
                        shutil.copy2(str(lbl_file), dst / f"labels/{split}" / lbl)
                    else:
                        open(dst / f"labels/{split}" / lbl, "w").close()

    copy_patient_files(train_patients, "train")
    copy_patient_files(val_patients, "val")
    copy_patient_files(test_patients, "test")

    return str(dst)


def write_data_yaml(dst_root, yaml_path, class_names):
    cfg = {
        "path": str(Path(dst_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)
    return yaml_path


def main():
    src_root = "patients" 
    dst_root = "dataset" 
    train_ratio = 0.7
    val_ratio = 0.15
    seed = 42
    yaml_out = "data.yaml"
    class_names = ["acetabular cup"]  
    out = prepare_by_patient(
        src_root, dst_root, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    # print(out)
    ypath = write_data_yaml(out, yaml_out, class_names)
    # print(ypath)

if __name__ == "__main__":
    main()
