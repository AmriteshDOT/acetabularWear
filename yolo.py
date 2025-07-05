import os
import yaml
from ultralytics import YOLO
# Data directories
TRAIN_IMG_DIR = "data/images/train"
TRAIN_LABEL_DIR = "data/labels/train"
VAL_IMG_DIR = "data/images/val"
VAL_LABEL_DIR = "data/labels/val"
TEST_IMG_DIR = "data/images/test"

# Model and training settings
PRETRAINED = "yolov8x.pt"       # checkpoint (yolov8n.pt, yolov8m.pt, yolov8x.pt)
FREEZE_LAYERS = [0, 1, 2]         # backbone stages to freeze
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
CONF_THRES = 0.25                # confidence threshold for inference
IOU_THRES = 0.50                 # IoU threshold for NMS/filtering
PROJECT = "runs/hip_cup_finetune"
RUN_NAME = "yolov8x_frozen"
NUM_CLASSES = 1
CLASS_NAMES = ["acetabular_cup"]
OUTPUT_DIR = "output"         

os.makedirs(OUTPUT_DIR, exist_ok=True)

# yaml YOLO
data_cfg = {
    'path': os.getcwd(),
    'train': os.path.relpath(TRAIN_IMG_DIR, os.getcwd()),
    'val': os.path.relpath(VAL_IMG_DIR, os.getcwd()),
    'nc': NUM_CLASSES,
    'names': CLASS_NAMES
}
with open('data.yaml', 'w') as f:
    yaml.dump(data_cfg, f)
print(yaml.dump(data_cfg))

# YOLO and freeze
model = YOLO(PRETRAINED)
model.model.freeze(FREEZE_LAYERS)


model.train(
    data='data.yaml',
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    freeze=FREEZE_LAYERS,
    project=PROJECT,
    name=RUN_NAME
)

best_weights = os.path.join(PROJECT, RUN_NAME, 'weights', 'best.pt')
best_model = YOLO(best_weights)
val_metrics = best_model.val(
    data='data.yaml',
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRES,
    iou=IOU_THRES
)
# validation metrics
for k, v in val_metrics.items():
    print(f" {k}: {v}")

# save annotated
best_model.predict(
    source=TEST_IMG_DIR,
    imgsz=IMG_SIZE,
    conf=CONF_THRES,
    iou=IOU_THRES,
    save=True,
    project=OUTPUT_DIR,
    name='test_predictions'
)

