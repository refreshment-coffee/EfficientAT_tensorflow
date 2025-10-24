import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorflow as tf
import numpy as np
class CedFeatureExtractorTF():
    def __init__(
        self,
        sample_rate=16000,
        win_size=512,
        hop_size=160,
        n_fft=512,
        n_mels=64,
        f_min=0.0,
        f_max=None,
        top_db=120.0,
        **kwargs
    ):

        self.sample_rate = sample_rate
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2.0
        self.top_db = top_db

        # Mel 矩阵和窗口函数初始化为变量，避免每次构造
        self.mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max
        )

    def __call__(self, waveform):
        # waveform: [T] or [B, T]
        if isinstance(waveform, np.ndarray):
            waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        if waveform.shape.rank == 1:
            waveform = tf.expand_dims(waveform, 0)  # [1, T]

        # Padding 类似 center=True
        pad = self.n_fft // 2
        waveform = tf.pad(waveform, [[0, 0], [pad, pad]], mode='REFLECT')

        # Hann window: 与 torchaudio 保持一致
        def window_fn(frame_length, dtype=tf.float32):
            return tf.signal.hann_window(frame_length, periodic=True, dtype=dtype)
        # STFT
        stfts = tf.signal.stft(
            signals=waveform,
            frame_length=self.win_size,
            frame_step=self.hop_size,
            fft_length=self.n_fft,
            window_fn= window_fn,
            pad_end=False
        )

        # 获取 magnitude
        magnitude_spectrograms = tf.abs(stfts) ** 2  # [B, T, F]

        # 计算 Mel 特征
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, self.mel_weight_matrix, axes=1)
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(self.n_mels))

        # 归一化为 dB (log10 转换，模拟 AmplitudeToDB)
        log_mel = 10.0 * tf.math.log(mel_spectrograms + 1e-10) / tf.math.log(10.0)

        # top_db clip，模仿 AmplitudeToDB 的行为
        max_log = tf.reduce_max(log_mel, axis=[1, 2], keepdims=True)
        log_mel_db = tf.clip_by_value(log_mel, clip_value_min=max_log - self.top_db, clip_value_max=max_log)

        # 调整维度 [B, mel, time]
        return tf.transpose(log_mel_db, perm=[0, 2, 1])


class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, samples_per_cls, num_classes, loss_type='softmax'):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.samples_per_cls = samples_per_cls
        self.num_classes = num_classes
        self.loss_type = loss_type

        # 预计算每类的权重
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_cls))
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights) * num_classes  # 归一化
        self.class_weights = weights.cuda()

    def forward(self, logits, labels):
        if self.loss_type == 'softmax':
            weights = self.class_weights[labels]  # 获取每个样本的类别权重
            loss = F.cross_entropy(logits, labels, reduction='none')  # 不加权先求每个样本的 loss
            loss = weights * loss
            return torch.mean(loss)
        else:
            raise NotImplementedError("Only 'softmax' is implemented.")


import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        if len(features.shape)==3:
            features=features.mean(axis=1)
        # print(features.shape)
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  ##同label

        # print("features.shape ",features.shape)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)# 避免自身对比
        mask = mask * logits_mask##非自身的同label

        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

