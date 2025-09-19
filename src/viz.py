import cv2
import os
from pathlib import Path
from ultralytics import YOLO


def save_annotated_images(model, img_list, out_dir, imgsz=640, conf=0.25, iou=0.45):
    os.makedirs(out_dir, exist_ok=True)
    res = model.predict(
        img_list, imgsz=imgsz, conf=conf, iou=iou, save=False, verbose=False
    )
    for r in res:
        ann = r.plot()
        img_path = r.path
        filename = Path(img_path).name
        out_path = Path(out_dir) / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))


def viz_random(img_root, preds_df, out_dir, class_names=None, n=5):
    os.makedirs(out_dir, exist_ok=True)
    if preds_df.empty:
        return
    sample = preds_df.sample(min(n, len(preds_df)))
    for _, r in sample.iterrows():
        p = Path(img_root) / r["img"]
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (
            f"{class_names[int(r['cls'])]} {r['conf']:.2f}"
            # if class_names
            # else f"{r['conf']:.2f}"
        )
        cv2.putText(
            img,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(Path(out_dir) / f"viz_{Path(r['img']).name}"), img)


def main():
    imgs_dir = "images" 
    out_dir = "out_viz" 
    mdl = "yolov8n.pt"  
    imgsz = 640
    conf = 0.25
    iou = 0.45

    y = YOLO(mdl)
    p = Path(imgs_dir)
    imgs = [str(x) for pat in ("*.jpg", "*.jpeg", "*.png") for x in p.glob(pat)]
    save_annotated_images(y, imgs, out_dir, imgsz=imgsz, conf=conf, iou=iou)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
