import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import librosa

from torch.utils.data import Dataset
import torch
import numpy as np
from functools import partial


class PreprocessDataset(Dataset):
    """A base preprocessing dataset representing a preprocessing step of a Dataset preprocessed on the fly.
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def get_roll_func(axis=1, shift=None, shift_range=4000):
    return partial(roll_func, axis=axis, shift=shift, shift_range=shift_range)


# roll waveform (over time)
def roll_func(b, axis=1, shift=None, shift_range=4000):
    x = b[0]
    others = b[1:]
    x = torch.as_tensor(x)
    sf = shift
    if shift is None:
        sf = int(np.random.random_integers(-shift_range, shift_range))
    return (x.roll(sf, axis), *others)


def get_gain_augment_func(gain_augment):
    return partial(gain_augment_func, gain_augment=gain_augment)


def gain_augment_func(b, gain_augment=12):
    x = b[0]
    others = b[1:]
    gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
    amp = 10 ** (gain / 20)
    x = x * amp
    return (x, *others)


# ------------- 用户自定义数据集目录 -------------

# SELECT_FOLDS=
# audio_dirs=SELECT_FOLDS
#
# print("-------------Success-------------")
#
#
# names=["Rain","Wind","Thunder","Bird","Forg","Insect","Dog","Honk","Background","Person","Other",
#        "Chaninsaw","Knock","Aerodynamic","Mech_work","Chanming","Cat","Firecracker","11Chaninsaw","114Engine",
#        "Chicken","Plane"]


# dataset_dir = r"/home/yons/Syz_dirs/Datasets"  # train_dataset 文件夹路径
# CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
#               '003Bird','004Frog','005Insect',
#               '006Dog','007Honk','008Background',
#               '009Person','010Other','011Drill',
#               '012Knock','013Aerodynamic','014Mech_work',
#               '015Chanming','016Cat','017Firecracker','111Chaninsaw','114Engine','018Chiken','019Plane'
#               ]

# CLASS2IDX = {name: idx for idx, name in enumerate(CUSTOM_CLASSES)}
# NUM_CLASSES = len(CUSTOM_CLASSES)


# ---------------- 辅助函数 ------------------
def pad_or_truncate(x, audio_length):
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[:audio_length]


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


# ---------------- Mixup ---------------------
class MixupDataset(TorchDataset):
    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, f1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, f2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, f1, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# ---------------- 数据集类 -----------------
class Custom22Dataset(TorchDataset):
    def __init__(self, dataset_dir, split="train", clip_length=2, resample_rate=32000, gain_augment=0,class2idx=None):
        """
        dataset_dir: train_dataset 文件夹
        split: "train" 或 "val"
        """
        self.dataset_dir = dataset_dir
        self.clip_length = clip_length * resample_rate
        self.resample_rate = resample_rate
        self.gain_augment = gain_augment
        self.class2idx=class2idx
        self.fold_names=[k for k in self.class2idx.keys()]
        # 遍历所有 fold
        all_samples = []
        for fold_name in self.fold_names:
            fold_path = os.path.join(dataset_dir, fold_name)
            if not os.path.isdir(fold_path):
                continue
            for fname in os.listdir(fold_path):
                if fname.endswith(".wav"):
                    # print(fold_name)
                    label = self.class2idx[fold_name]
                    all_samples.append({"path": os.path.join(fold_path, fname), "label": label})

        # 打乱顺序
        random.shuffle(all_samples)
        n_train = int(len(all_samples) * 0.9)
        if split == "train":
            self.samples = all_samples[:n_train]
        else:
            self.samples = all_samples[n_train:]
        print(f"{split} dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, _ = librosa.load(sample["path"], sr=self.resample_rate, mono=True)
        if self.gain_augment:
            waveform = pydub_augment(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, self.clip_length)
        # target = np.zeros(NUM_CLASSES, dtype=np.float32)
        target=sample["label"]
        return waveform.reshape(1, -1), sample["path"], target


# ---------------- Dataset 获取接口 -----------------
def get_custom22_train_set(dataset_dir="",clip_length=2,resample_rate=32000, roll=False, wavmix=False, gain_augment=0,class2idx=None):
    ds = Custom22Dataset(dataset_dir=dataset_dir,clip_length=clip_length, split="train", resample_rate=resample_rate, gain_augment=gain_augment,class2idx=class2idx)
    # if roll:
    #     ds = PreprocessDataset(ds, get_roll_func())
    # if wavmix:
    #     ds = MixupDataset(ds)
    return ds


def get_custom22_valid_set(dataset_dir="",clip_length=2,resample_rate=32000,class2idx=None):
    ds = Custom22Dataset(dataset_dir=dataset_dir, clip_length=clip_length,split="val", resample_rate=resample_rate, gain_augment=0,class2idx=class2idx)
    return ds
