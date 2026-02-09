# Video Feature Extraction and Cross-Subject Classification (Swin3D)

This project implements a pipeline for video feature extraction using a
pretrained Swin3D-B model and evaluates performance under cross-subject
conditions.

The goal is to extract meaningful video representations and assess
generalization to unseen subjects using GroupKFold validation.

---

## Pipeline

1. Load video files (.mp4)
2. Temporal sampling (16 frames per video)
3. Spatial preprocessing (resize to 224×224 + normalization)
4. Feature extraction using pretrained Swin3D (Kinetics-400)
5. Save embeddings (.pt)
6. Train a classifier on extracted features
7. Evaluate using GroupKFold (subject-wise split)

---

## Model

Feature extractor:

- Architecture: Swin3D-B
- Pretrained on: Kinetics-400
- Final classification head removed
- Output embedding size: 1024

Classifier:

- Linear(1024 → 256)
- ReLU
- Dropout(0.3)
- Linear(256 → num_classes)

---

## Dataset Structure

transfer_files/
├── \*.mp4  
├── labels_lesson_CV.txt  
└── features/ (generated embeddings)

Labels file format:

video_name,label_id

Subject ID is extracted from filename:
subject = video*name.split("*")[0]

---

## Installation

Create environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install torch torchvision pandas scikit-learn av

PyAV is required for video decoding.

---

## Usage

Run the full pipeline:

python lab3.py

Outputs:

- Feature files saved to transfer_files/features/
- Cross-validation accuracy printed for each fold

---

## Evaluation

Validation method: GroupKFold (5 folds)

Groups = subject IDs

This ensures:

- No subject overlap between train and test
- No identity leakage
- Evaluation of cross-subject generalization

---

## Results

Fold 0: 0.000  
Fold 1: 0.000  
Fold 2: 0.375  
Fold 3: 0.286  
Fold 4: 0.143

Mean accuracy ≈ 0.16

Low performance is expected due to:

- Small dataset
- Cross-subject distribution shift
- Frozen feature extractor (no fine-tuning)

---

## Environment

- Python 3.10
- PyTorch
- GPU recommended
- Tested on HPC environment (Grid5000)

---

## Author

Zeev Weizmann  
MSc Data Science & AI  
Université Côte d’Azur

## Lab Links

Page  
https://zeevweizmann.github.io/Feature-Extraction-with-Video-Swin-Transformer/

sSource Code  
https://github.com/ZeevWeizmann/Feature-Extraction-with-Video-Swin-Transformer

---
