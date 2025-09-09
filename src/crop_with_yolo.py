import os
from pathlib import Path
import cv2
from ultralytics import YOLO

# -------- CONFIG --------
MODEL_WEIGHTS = "runs/hip_cup_finetune/yolov8x_frozen/weights/best.pt"
SOURCE_DIR = "frames_by_patient"  # unannotated frames grouped by patient
OUTPUT_DIR = "crops_by_patient"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.50
# ------------------------


def load_model(weights):
    return YOLO(weights)


def crop_and_save(model, source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = [p for p in source_dir.iterdir() if p.is_dir()]
    for patient in patients:
        patient_out = output_dir / patient.name
        patient_out.mkdir(parents=True, exist_ok=True)

        images = list(patient.glob("*.jpg"))
        for img_path in images:
            results = model.predict(
                source=str(img_path),
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                save=False,
                verbose=False,
            )
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                im = cv2.imread(str(img_path))
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    crop = im[y1:y2, x1:x2]
                    out_name = f"{img_path.stem}_crop{i}.jpg"
                    out_path = patient_out / out_name
                    cv2.imwrite(str(out_path), crop)
        print(f"Processed {patient.name}, saved crops to {patient_out}")


if __name__ == "__main__":
    model = load_model(MODEL_WEIGHTS)
    crop_and_save(model, SOURCE_DIR, OUTPUT_DIR)
