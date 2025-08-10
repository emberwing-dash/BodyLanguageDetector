import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ===============================
# Config
# ===============================
CSV_PATH = r"D:\DataSet\MELD.Raw\train\train_sent_emo.csv"
VIDEO_DIR = r"D:\DataSet\MELD.Raw\train\train_splits"
N_FRAMES = 8
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Dataset with balanced sampling
# ===============================
class MELDVideoDataset(Dataset):
    def __init__(self, csv_path, video_dir, transform=None, n_frames=8):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.transform = transform
        self.n_frames = n_frames
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data["Emotion"].unique()))}

        # Group indices by emotion
        grouped = defaultdict(list)
        for i, row in self.data.iterrows():
            grouped[row["Emotion"]].append(i)

        min_count = min(len(idxs) for idxs in grouped.values())
        balanced_indices = []
        for emotion, idxs in grouped.items():
            balanced_indices.extend(np.random.choice(idxs, min_count, replace=False))
        self.indices = balanced_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[self.indices[idx]]
            video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_name)

            frames = self._load_video_frames(video_path, self.n_frames)
            if frames is None:
                raise RuntimeError("Empty frames")
            label = self.label2idx[row["Emotion"]]
            return frames, label

        except Exception as e:
            print(f"[WARNING] Skipping broken video: {video_path} ({e})")
            return None

    def _load_video_frames(self, path, num_frames):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None

        frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_idxs:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            idx += 1

        cap.release()
        if len(frames) == 0:
            return None
        while len(frames) < num_frames:
            frames.append(frames[-1])

        return torch.stack(frames)

# ===============================
# Model
# ===============================
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.cnn_out_dim = cnn.fc.in_features

        self.lstm = nn.LSTM(self.cnn_out_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        cnn_features = []
        for t in range(T):
            f = self.cnn(x[:, t])  # [B, D, 1, 1]
            f = f.view(B, -1)
            cnn_features.append(f)
        feats = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(feats)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ===============================
# Collate fn to skip None batches
# ===============================
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ===============================
# Transforms
# ===============================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ===============================
# Training Loop
# ===============================
if __name__ == "__main__":
    dataset = MELDVideoDataset(CSV_PATH, VIDEO_DIR, transform=transform, n_frames=N_FRAMES)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True,
                            collate_fn=collate_skip_none)

    num_classes = len(dataset.label2idx)
    model = CNN_LSTM(hidden_dim=256, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            if batch is None:
                continue
            frames, labels = batch
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "cnn_lstm_meld_balanced.pth")
    print("Training complete and model saved.")
