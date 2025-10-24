import torch.nn as nn
import torchaudio
import torch


class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec



import librosa

import torch
import torchaudio
import torch.nn as nn
import numpy as np
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[1], 'GPU')

# 让 TF 不抢占全部 GPU 显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow memory growth enabled.")
    except RuntimeError as e:
        print(e)


class AugmentMelSTFTTF:  ##SYZ  pytorch语句转tensroflow语句
    def __init__(self,
                 sr=32000,
                 n_fft=1024,
                 hopsize=320,
                 win_length=800,
                 n_mels=128,
                 fmin=0.0,
                 fmax=15000):
        self.sr = sr
        self.n_fft = n_fft
        self.hopsize = hopsize
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # mel 权重矩阵可以提前计算好
        self.mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sr,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax
        )

    def __call__(self, waveform):
        """
        waveform: [batch, time], tf.Tensor
        返回: [batch, time_frames, n_mels]
        """
        # 预加重
        waveform = tf.concat([waveform[:, :1],
                              waveform[:, 1:] - 0.97 * waveform[:, :-1]], axis=1)

        # STFT
        stft = tf.signal.stft(
            waveform,
            frame_length=self.win_length,
            frame_step=self.hopsize,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
            pad_end=True
        )
        power_spec = tf.abs(stft) ** 2

        # mel 滤波
        mel_spec = tf.matmul(power_spec, self.mel_weight)

        # log
        mel_spec = tf.math.log(mel_spec + 1e-5)

        # fast normalization
        mel_spec = (mel_spec + 4.5) / 5.

        mel_spec=np.transpose(mel_spec,axes=(0,2,1))
        return mel_spec

