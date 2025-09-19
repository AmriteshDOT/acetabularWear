import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd


def save_crops(img_root, preds_df, out_root, pad=0.15, size=224):
    os.makedirs(out_root, exist_ok=True)
    rows = []
    for idx in range(len(preds_df)):
        r = preds_df.iloc[idx]
        p = Path(img_root) / r["img"]
        if not p.exists():
            found = list(Path(img_root).rglob(r["img"]))
            if found:
                p = found[0]
            else:
                continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        x1 = int(max(0, r["x1"]))
        y1 = int(max(0, r["y1"]))
        x2 = int(min(w - 1, r["x2"]))
        y2 = int(min(h - 1, r["y2"]))
        ww = x2 - x1
        hh = y2 - y1
        x1p = int(max(0, x1 - pad * ww))
        x2p = int(min(w - 1, x2 + pad * ww))
        y1p = int(max(0, y1 - pad * hh))
        y2p = int(min(h - 1, y2 + pad * hh))
        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        h2, w2 = crop.shape[:2]
        scale = size / max(h2, w2)
        nh, nw = int(h2 * scale), int(w2 * scale)
        crop = cv2.resize(crop, (nw, nh))
        pad_img = 255 * np.ones((size, size, 3), dtype=crop.dtype)
        pad_img[:nh, :nw] = crop
        outp = Path(out_root) / f"{Path(r['img']).stem}_c{int(r['cls'])}_{idx}.jpg"
        cv2.imwrite(str(outp), pad_img)
        rows.append(
            {
                "crop": str(outp),
                "src": r["img"],
                "cls": int(r["cls"]),
                "conf": float(r["conf"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    img_root = "images"    
    out_root = "crops"      
    pad = 0.15
    size = 224

    preds_csv = "preds.csv"
    import pandas as pd
    preds_df = pd.read_csv(preds_csv)

    df = save_crops(img_root, preds_df, out_root, pad=pad, size=size)
    # print(f"saved {len(df)} crops : {out_root}")
    df.to_csv(Path(out_root) / "crops_log.csv", index=False)


if __name__ == "__main__":
    main()
