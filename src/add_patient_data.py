import os
import shutil
from pathlib import Path

# ---------- CONFIG ----------
ANNOTATED_DIR = "annotated_crops_by_patient"
UNANNOTATED_DIR = "crops_by_patient"
OUTPUT_DIR = "combined_crops_by_patient"
# ----------------------------


def add_patient_images(annotated_dir, unannotated_dir, output_dir):
    annotated_dir = Path(annotated_dir)
    unannotated_dir = Path(unannotated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = set()
    if annotated_dir.exists():
        patients.update([p.name for p in annotated_dir.iterdir() if p.is_dir()])
    if unannotated_dir.exists():
        patients.update([p.name for p in unannotated_dir.iterdir() if p.is_dir()])

    for patient_id in sorted(patients):
        patient_out = output_dir / patient_id
        patient_out.mkdir(parents=True, exist_ok=True)

        # Copy from annotated
        if (annotated_dir / patient_id).exists():
            for f in (annotated_dir / patient_id).glob("*.jpg"):
                shutil.copy(f, patient_out / f"annot_{f.name}")

        # Copy from unannotated
        if (unannotated_dir / patient_id).exists():
            for f in (unannotated_dir / patient_id).glob("*.jpg"):
                shutil.copy(f, patient_out / f"unann_{f.name}")

        print(f"Patient {patient_id}: merged into {patient_out}")


if __name__ == "__main__":
    add_patient_images(ANNOTATED_DIR, UNANNOTATED_DIR, OUTPUT_DIR)
