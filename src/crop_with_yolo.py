import os
from pathlib import Path
import cv2
from ultralytics import YOLO

weights_path = "runs/hip_cup_finetune/yolov8x_frozen/weights/best.pt"
frames_dir = "frames_by_patient" 
crops_dir = "crops_by_patient"
img_size = 640
conf_thres = 0.25
iou_thres = 0.50

def load_model(weights):
    return YOLO(weights)

def crop_frames(model, source, out_root):
    source = Path(source)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    patients = [p for p in source.iterdir() if p.is_dir()]
    for patient in patients:
        patient_out = out_root / patient.name
        patient_out.mkdir(parents=True, exist_ok=True)

        for img_path in patient.glob("*.jpg"):
            results = model.predict(
                source=str(img_path),
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                save=False,
                verbose=False
            )
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                img = cv2.imread(str(img_path))
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    crop = img[y1:y2, x1:x2]
                    out_name = f"{img_path.stem}_crop{i}.jpg"
                    cv2.imwrite(str(patient_out / out_name), crop)

        print(f"{patient.name}: crops saved -> {patient_out}")

if __name__ == "__main__":
    model = load_model(weights_path)
    crop_frames(model, frames_dir, crops_dir)
