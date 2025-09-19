from pathlib import Path
from ultralytics import YOLO

def train_yolo(yaml_p, pretrained, proj, name, epochs=20, imgsz=640, batch=8, freeze=4):
    m = YOLO(pretrained)
    m.freeze = range(len(m.model) - freeze)
    m.train(
        data=yaml_p, epochs=epochs, imgsz=imgsz, batch=batch, project=proj, name=name
    )
    return str(Path(proj) / name / "weights")


def pick_best(weights_dir):
    p = Path(weights_dir)
    b = p / "best.pt"
    if b.exists():
        return str(b)
    l = p / "last.pt"
    if l.exists():
        return str(l)
    return None


def main():
    yaml_p = "data.yaml"
    pretrained = "yolov8n.pt"
    proj = "runs"
    name = "cup_run"
    epochs = 20
    imgsz = 640
    batch = 8
    freeze = 4
    # train
    wdir = train_yolo(yaml_p, pretrained, proj, name, epochs=epochs, imgsz=imgsz, batch=batch, freeze=freeze)
    # print(wdir)
    best = pick_best(Path(wdir) / "weights")
    # print(best)


if __name__ == "__main__":
    main()
