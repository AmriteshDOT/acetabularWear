import os
import cv2
from ultralytics import YOLO

# 1. Load your fine-tuned YOLO model
model = YOLO('runs/hip_cup_finetune/yolov8x_frozen/weights/best.pt')

# 2. Where your test/original images live
IMAGE_DIR  = 'data/images/test'
OUTPUT_DIR = 'cropped_boxes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Loop over images
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # 4. Run inference
    results = model(img)  # returns a list of Results; here length=1

    # 5. Iterate over all detected boxes in this image
    for i, box in enumerate(results[0].boxes.xyxy):  # xyxy = [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img[y1:y2, x1:x2]
        # save
        base, ext = os.path.splitext(img_name)
        out_name = f"{base}_det{i}{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), crop)

