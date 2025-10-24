# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
import random
import numpy as np
from efficientat.model import EfficientAT

# ---------------------------
# 配置
# ---------------------------
DATA_DIR = "./data"  # 数据目录
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
LABELS = ["Rain","Wind","Thunder","Bird","Forg","Insect","Dog","Honk","Background","Person","Other",
          "Chaninsaw","Knock","Aerodynamic","Mech_work","Chanming","Cat","Firecracker","11Chaninsaw",
          "114Engine","Chicken","Plane"]
NUM_CLASSES = len(LABELS)
SAMPLE_RATE = 32000
DURATION = 10  # 秒
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
FREEZE_BACKBONE = True  # 是否冻结前面层，只训练分类头

# ---------------------------
# 数据集
# ---------------------------
class AudioDataset(Dataset):
    def __init__(self, root_dir, labels, sr=32000, duration=10, augment=False):
        self.samples = []
        self.sr = sr
        self.duration = duration
        self.max_len = sr * duration
        self.augment = augment
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=320, n_mels=128, f_min=50, f_max=14000
        )
        for label_idx, label in enumerate(labels):
            label_dir = os.path.join(root_dir, label)
            if not os.path.exists(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".wav"):
                    self.samples.append((os.path.join(label_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)
        waveform = torchaudio.functional.resample(waveform, sr, self.sr)

        # 处理长度
        if waveform.shape[1] < self.max_len:
            pad_len = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.shape[1] > self.max_len:
            start = random.randint(0, waveform.shape[1] - self.max_len)
            waveform = waveform[:, start:start + self.max_len]

        mel = self.mel(waveform).log()
        return mel, label

# ---------------------------
# 数据加载
# ---------------------------
train_dataset = AudioDataset(TRAIN_DIR, LABELS, sr=SAMPLE_RATE, duration=DURATION)
val_dataset = AudioDataset(VAL_DIR, LABELS, sr=SAMPLE_RATE, duration=DURATION)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

# ---------------------------
# 模型
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientAT.from_pretrained("EfficientAT_A2")
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(device)

if FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

# ---------------------------
# 损失和优化器
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ---------------------------
# 训练循环
# ---------------------------
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for mel, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        mel, labels_batch = mel.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

    # ---------------------------
    # 验证
    # ---------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, labels_batch in val_loader:
            mel, labels_batch = mel.to(device), labels_batch.to(device)
            outputs = model(mel)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4%}")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "efficientat_best.pth")
        print("Saved Best Model.")

print("Training complete! Best validation accuracy:", best_acc)
