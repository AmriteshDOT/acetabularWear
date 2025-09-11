import os
import yaml
from ultralytics import YOLO

#paths
train_images = "data/images/train"
train_labels = "data/labels/train"
val_images = "data/images/val"
val_labels = "data/labels/val"
test_images = "data/images/test"

#config
pretrained_weights = "yolov8x.pt"
freeze_layers = [0, 1, 2]
epochs = 50
img_size = 640
batch_size = 8

#thresholds
conf_thresh = 0.25
iou_thresh = 0.50

# output
project_dir = "runs/hip_cup_finetune"
run_name = "yolov8x_frozen"
num_classes = 1
class_names = ["acetabular_cup"]
out_dir = "output"
data_yaml_path = "data.yaml"


def write_data_yaml():
    cfg = {
        "path": os.getcwd(),
        "train": os.path.relpath(train_images, os.getcwd()),
        "val": os.path.relpath(val_images, os.getcwd()),
        "nc": num_classes,
        "names": class_names,
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(cfg, f)
    print("Wrote", data_yaml_path)
    print(yaml.dump(cfg))
    return data_yaml_path


def train_model(data_yaml, pretrained, freeze, project, name):
    model = YOLO(pretrained)
    model.model.freeze(freeze)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        freeze=freeze,
        project=project,
        name=name,
    )
    return os.path.join(project, name, "weights", "best.pt")


def validate(weights, data_yaml):
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        conf=conf_thresh,
        iou=iou_thresh,
    )
    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f" {k}: {v}")
    return model


def inference_and_save(model, source, output, run_name):
    model.predict(
        source=source,
        imgsz=img_size,
        conf=conf_thresh,
        iou=iou_thresh,
        save=True,
        project=output,
        name=run_name,
    )
    print(f"\nSaved predictions to {output}/{run_name}")


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    data_yaml = write_data_yaml()
    best_weights_path = train_model(
        data_yaml, pretrained_weights, freeze_layers, project_dir, run_name
    )
    best_model = validate(best_weights_path, data_yaml)
    inference_and_save(best_model, test_images, out_dir, "test_predictions")
