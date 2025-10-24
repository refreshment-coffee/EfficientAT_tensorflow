# train.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from pathlib import Path

# -----------------------------
# 自定义 Dataset
# -----------------------------
class MyAudioDataset(Dataset):
    """
    Dataset for your 22-class audio dataset
    Assumes:
      - audio files in audio_path/{class_name}/*.wav
      - class_names is list of 22 class names in order
    """
    def __init__(self, audio_path, class_names, transform=None, sample_rate=32000, duration=4.0):
        self.audio_path = Path(audio_path)
        self.class_names = class_names
        self.transform = transform
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.data = []

        for idx, cls_name in enumerate(class_names):
            cls_dir = self.audio_path / cls_name
            if not cls_dir.exists():
                continue
            for file in cls_dir.glob("*.wav"):
                self.data.append({
                    "path": str(file),
                    "label": idx
                })

        print(f"Loaded {len(self.data)} samples from {len(class_names)} classes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform, sr = torchaudio.load(item["path"])
        waveform = waveform.mean(0)  # convert to mono

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Pad or truncate to fixed length
        if waveform.size(0) < self.num_samples:
            repeats = int(np.ceil(self.num_samples / waveform.size(0)))
            waveform = waveform.repeat(repeats)[:self.num_samples]
        else:
            # random crop
            start = random.randint(0, waveform.size(0) - self.num_samples)
            waveform = waveform[start:start+self.num_samples]

        if self.transform:
            waveform = self.transform(waveform)

        label = item["label"]
        return waveform.float(), label

# -----------------------------
# 构建模型
# -----------------------------
def build_model(pretrained=True, model_name="mn10_as", num_classes=22, freeze_backbone=False):
    """
    Load EfficientAT from official repo, modify final head for custom classes
    """
    from efficientat.models import EfficientAT  # assuming repo is cloned and in PYTHONPATH

    model = EfficientAT(model_name=model_name, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    return model

# -----------------------------
# 训练函数
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveform, labels in loader:
        waveform = waveform.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveform)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveform.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += waveform.size(0)

    return running_loss / total, correct / total

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveform, labels in loader:
            waveform = waveform.to(device)
            labels = labels.to(device)
            logits = model(waveform)
            loss = criterion(logits, labels)

            running_loss += loss.item() * waveform.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += waveform.size(0)

    return running_loss / total, correct / total

# -----------------------------
# 主训练脚本
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    class_names = args.class_names.split(",")
    train_dataset = MyAudioDataset(args.train_path, class_names, sample_rate=args.sample_rate, duration=args.duration)
    val_dataset   = MyAudioDataset(args.val_path, class_names, sample_rate=args.sample_rate, duration=args.duration)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(pretrained=args.pretrained, model_name=args.model_name, num_classes=len(class_names), freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="训练集路径")
    parser.add_argument("--val_path", type=str, required=True, help="验证集路径")
    parser.add_argument("--class_names", type=str, required=True, help="类别名称,用逗号分隔")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=4.0, help="训练音频裁剪时长(s)")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model_name", type=str, default="mn10_as")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--save_path", type=str, default="./efficientat_best.pth")
    args = parser.parse_args()

    main(args)
