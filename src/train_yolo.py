import os
import yaml
from ultralytics import YOLO

# ---------- CONFIG ----------
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"
VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"
TEST_IMG_DIR = "data/images/test"

PRETRAINED = "yolov8x.pt"
FREEZE_LAYERS = [0, 1, 2]
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
CONF_THRES = 0.25
IOU_THRES = 0.50
PROJECT = "runs/hip_cup_finetune"
RUN_NAME = "yolov8x_frozen"
NUM_CLASSES = 1
CLASS_NAMES = ["acetabular_cup"]
OUTPUT_DIR = "output"
DATA_YAML = "data.yaml"
# ----------------------------


def make_data_yaml():
    cfg = {
        "path": os.getcwd(),
        "train": os.path.relpath(TRAIN_IMG_DIR, os.getcwd()),
        "val": os.path.relpath(VAL_IMG_DIR, os.getcwd()),
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f)
    print("Wrote data.yaml:")
    print(yaml.dump(cfg))
    return DATA_YAML


def train_yolo(data_yaml, pretrained, freeze_layers, project, run_name):
    model = YOLO(pretrained)
    model.model.freeze(freeze_layers)

    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        freeze=freeze_layers,
        project=project,
        name=run_name,
    )
    return os.path.join(project, run_name, "weights", "best.pt")


def validate_model(weights, data_yaml):
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE, conf=CONF_THRES, iou=IOU_THRES
    )
    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f" {k}: {v}")
    return model


def run_inference(model, source, output_dir, name):
    model.predict(
        source=source,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        save=True,
        project=output_dir,
        name=name,
    )
    print(f"\nSaved test predictions to {output_dir}/{name}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_yaml = make_data_yaml()
    best_weights = train_yolo(data_yaml, PRETRAINED, FREEZE_LAYERS, PROJECT, RUN_NAME)
    best_model = validate_model(best_weights, data_yaml)
    run_inference(best_model, TEST_IMG_DIR, OUTPUT_DIR, "test_predictions")
