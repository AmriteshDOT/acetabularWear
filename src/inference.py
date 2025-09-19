import pandas as pd
from pathlib import Path
import json


def batch_infer(model, img_list, imgsz=640, conf=0.25, iou=0.45):
    recs = []
    res = model.predict(img_list, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    for r in res:
        name = Path(getattr(r, "path", getattr(r, "orig_path", None))).name
        if hasattr(r, "boxes") and len(r.boxes):
            bb = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(bb)):
                x1, y1, x2, y2 = bb[i]
                recs.append(
                    {
                        "img": name,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "cls": int(cls[i]),
                        "conf": float(confs[i]),
                    }
                )
    return recs


def save_preds_csv(recs, out_csv):
    df = pd.DataFrame(recs)
    if df.empty:
        df = pd.DataFrame(columns=["img", "x1", "y1", "x2", "y2", "cls", "conf"])
    df.to_csv(out_csv, index=False)
    return df


def evaluate_model(
    weights,
    data_yaml,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    splits=("val", "test"),
    save_dir=None,
):
    from ultralytics import YOLO 

    out = {}
    model = YOLO(weights)
    d = Path(save_dir or ".")
    d.mkdir(parents=True, exist_ok=True)

    for s in splits:
        res = model.val(
            data=data_yaml, split=s, imgsz=imgsz, conf=conf, iou=iou, save_json=True
        )
        box = getattr(res, "box", None)
        metrics = {
            "map50:95": float(getattr(box, "map", float("nan"))) if box else None,
            "map50": float(getattr(box, "map50", float("nan"))) if box else None,
        }
        out[s] = metrics
        j = res.to_json()
        (d / f"val_{s}_results.json").write_text(j)
    if save_dir:
        (d / "metrics_summary.json").write_text(json.dumps(out, indent=2))

    return out


def main():
    weights = "yolov8n.pt"
    data_yaml = "data.yaml"
    imgsz = 640
    conf = 0.25
    iou = 0.45
    splits = ("val", "test")
    save_dir = "eval_results"
    out = evaluate_model(
        weights,
        data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        splits=splits,
        save_dir=save_dir,
    )
    # print(save_dir)
    # print(out)


if __name__ == "__main__":
    main()
