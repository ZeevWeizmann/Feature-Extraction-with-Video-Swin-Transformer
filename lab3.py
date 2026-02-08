import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from torchvision.io import read_video
import torchvision.transforms as T
from torchvision.models.video import swin3d_b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(os.path.expanduser("~/transfer_files"))


VIDEO_DIR = BASE_DIR
LABELS_PATH = BASE_DIR / "labels_lesson_CV.txt"
FEAT_DIR = BASE_DIR / "features"
FEAT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(LABELS_PATH)
df["subject"] = df["video_name"].str.split("_").str[0]

labels = sorted(df.label_id.unique())
label2idx = {label: i for i, label in enumerate(labels)}
df["label_idx"] = df["label_id"].map(label2idx)

video_model = swin3d_b(weights="KINETICS400_V1")
video_model.head = nn.Identity()
video_model = video_model.to(device)
video_model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_video(path):
    video, _, _ = read_video(str(path), pts_unit="sec")
    return video

def temporal_sample(video, num_frames=16):
    T_len = video.shape[0]
    if T_len >= num_frames:
        idx = torch.linspace(0, T_len - 1, num_frames).long()
        video = video[idx]
    else:
        pad = video[-1:].repeat(num_frames - T_len, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)
    return video

def preprocess(video):
    video = video.float() / 255.0
    video = video.permute(0, 3, 1, 2)
    video = torch.stack([transform(f) for f in video])
    video = video.permute(1, 0, 2, 3)
    video = video.unsqueeze(0)
    return video

@torch.no_grad()
def extract_features(video_path):
    video = load_video(video_path)
    video = temporal_sample(video)
    video = preprocess(video).to(device)
    feat = video_model(video)
    return feat.squeeze(0).cpu()

for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
    out = FEAT_DIR / (mp4.stem + ".pt")
    if out.exists():
        continue

    print("Processing:", mp4)

    try:
        feat = extract_features(mp4)
        torch.save(feat, out)
    except Exception as e:
        print("ERROR with:", mp4)
        print(e)
        break

X = []
y = []
subjects = []

for _, row in df.iterrows():
    feat_path = FEAT_DIR / row.video_name.replace(".mp4", ".pt")
    if not feat_path.exists():
        continue
    X.append(torch.load(feat_path))
    y.append(row.label_idx)
    subjects.append(row.subject)

X = torch.stack(X)
y = torch.tensor(y)
subjects = pd.Series(subjects)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

gkf = GroupKFold(n_splits=5)
num_classes = len(label2idx)

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=subjects)
):
    model = Classifier(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)

    acc = accuracy_score(y_test.cpu(), preds.cpu())
    print(f"Fold {fold} Accuracy: {acc:.3f}")
s