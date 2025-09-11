import os
import shutil
from pathlib import Path

annotated = "annoPatient"
unannotated = "unannoPatient"
output = "combined"


def add_patient_images(annotated, unannotated, output):
    annotated = Path(annotated)
    unannotated = Path(unannotated)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    patients = set()
    if annotated.exists():
        patients.update([p.name for p in annotated.iterdir() if p.is_dir()])
    if unannotated.exists():
        patients.update([p.name for p in unannotated.iterdir() if p.is_dir()])

    for p_id in sorted(patients):
        p_out = output / p_id
        p_out.mkdir(parents=True, exist_ok=True)

        #from annotated
        if (annotated / p_id).exists():
            for f in (annotated / p_id).glob("*.jpg"):
                shutil.copy(f, p_out / f"annot_{f.name}")

        #from unannotated
        if (unannotated / p_id).exists():
            for f in (unannotated / p_id).glob("*.jpg"):
                shutil.copy(f, p_out / f"unann_{f.name}")

        print(f"Patient {p_id}: merged into {p_out}")


if __name__ == "__main__":
    add_patient_images(annotated, unannotated, output)
