
## Hip Joint Prosthesis Wear Severity Prediction
- This project implements a two-stage deep learning pipeline to detect and classify acetabular cup wear severity (hip joint prosthesis wear) into four categories: __Normal, Mild, Moderate, Severe.__
- Helps doctors with triage and post-operative monitoring.
- Generalizes across all severity levels of prosthesis wear.
- Developed during my Undergraduate Research Internship.

## Pipeline Overview
#### Hip Region Localization
- YOLOv8 used to detect and localize hip joint prosthesis in X-ray/MRI frames.
#### Wear Severity Classification
- Fine-tuned CNN with DenseNet121 backbone for classifying wear severity into 4 classes.
## Dataset & Preprocessing
- Frames extracted from medical imaging videos (X-rays/MRI).
- Controlled data augmentation applied to improve generalization.
- Balanced dataset prepared for classification.
## Training
- Optimized hyperparameters using Optuna.
- Trained YOLOv8 for detection and DenseNet121 for classification.
## Results
- mAP@ 0.5:0.95 (Detection): 0.80
- Classification Accuracy: 0.84
- Macro F1-score: 0.79
## Tech Stack
- YOLOv8 (Ultralytics) – for detection
- PyTorch – for CNN classification
- Optuna – for hyperparameter optimization
- OpenCV, NumPy, Pandas – for preprocessing