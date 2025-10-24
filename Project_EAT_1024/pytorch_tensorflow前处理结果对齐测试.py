import librosa

audio_path="./test_audios/0-rain025.WAV"

##tensorflow


import torch
import torchaudio
import torch.nn as nn
import numpy as np
import tensorflow as tf

import torch.nn as nn
class AugmentMelSTFT2(nn.Module):
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


    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        # fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        # fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        # if not self.training:
        fmin = self.fmin
        fmax = self.fmax
        print("fmin: {} , fmax: {} ".format(fmin, fmax))
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()



        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec
sample_rate=32000
target_sec=2
def get_wavform(audio_path):
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    ###固定长度 2s

    target_len = int(sample_rate * target_sec)
    cur_len = len(waveform)

    # 过短 补长
    if cur_len < target_len:
        pad_len = target_len - cur_len
            # 在尾部补零
        waveform = np.pad(waveform, (0, pad_len), mode="constant")


    elif cur_len > target_len:

        start = (cur_len - target_len) // 2
        waveform = waveform[start:start + target_len]
    return waveform

    # return waveform.astype(np.float32)




    # print(waveform[:10])




# ==============================
# 1️⃣ PyTorch 版本（原样保留）
# ==============================
class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
                 freqm=48, timem=192, fmin=0.0, fmax=None,
                 fmin_aug_range=10, fmax_aug_range=2000):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax}")
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer('window', torch.hann_window(win_length, periodic=False), persistent=False)

        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x):
        # 预加重
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        # STFT
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)

        # 动态频率增强范围
        # fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        # fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()


        # if not self.training:  # eval 模式下不做增强
        fmin, fmax = self.fmin, self.fmax
        print("fmin: {} , fmax: {} ".format(fmin, fmax))
        # Mel 滤波器
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr,
            fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        # log
        melspec = (melspec + 1e-5).log()



        # fast normalization
        melspec = (melspec + 4.5) / 5.
        return melspec


# ==============================
# 2️⃣ TensorFlow 对齐实现
# ==============================
def tf_augment_melstft(waveform, sr=32000, n_fft=1024, hopsize=320, win_length=800,
                       n_mels=128, fmin=0.0, fmax=15000,
                       training=False, freqm=0, timem=0):
    # 预加重
    waveform = tf.concat([waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], axis=1)

    # STFT
    stft = tf.signal.stft(
        waveform,
        frame_length=win_length,
        frame_step=hopsize,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    power_spec = tf.abs(stft) ** 2

    # mel 滤波器
    mel_weight = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
        # mel_scale='htk'  # ← Kaldi 默认使用 HTK 标准
    )
    mel_spec = tf.matmul(power_spec, mel_weight)

    # log
    mel_spec = tf.math.log(mel_spec + 1e-5)

    # fast normalization
    mel_spec = (mel_spec + 4.5) / 5.

    return mel_spec


import tensorflow as tf

class AugmentMelSTFTTF:
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

        return mel_spec

# ==============================
# 3️⃣ 对齐测试
# ==============================
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # 随机输入波形
    waveform = get_wavform(audio_path)
    waveform_torch = torch.from_numpy(waveform).unsqueeze(0)
    waveform_tf = tf.convert_to_tensor(waveform[None, :])

    # 固定 fmin/fmax 确保可对齐
    fmin, fmax = 0.0, 15000.0

    # === PyTorch ===
    # model = AugmentMelSTFT(sr=sr, fmax=fmax)

    ##pytorch

    n_mels = 128
    sample_rate = 32000
    window_size = 800
    hop_size = 320
    target_sec = 2
    fmax=15000
    cuda = True

    print("AugmentMelSTFT")
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size, fmax=15000)
    mel_torch = mel(waveform_torch).cpu().numpy()


    mel2=AugmentMelSTFT2(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size, fmax=15000)
    mel_torch2=mel2(waveform_torch).cpu().numpy()


    print("tf_augment_melstft")
    # === TensorFlow ===
    mel_tf = tf_augment_melstft(waveform_tf, sr=sample_rate, fmax=fmax, training=False).numpy().T  # 对齐维度
    mel_tf = tf.transpose(mel_tf, perm=[2,0,1])  # [batch, n_mels, time]

    print("mel_torch.shape: ",np.array(mel_torch).shape)
    print("mel_tf.shape: ", np.array(mel_tf).shape)

    mel_tf2=AugmentMelSTFTTF(sr=sample_rate, fmax=fmax)
    mel_tf_output2=mel_tf2(waveform_tf)
    mel_tf_output2 = tf.transpose(mel_tf_output2, perm=[0, 2, 1])
    print("mel_tf_output2.shape: ",np.array(mel_tf_output2).shape)

    print("calculate diff")
    print(mel_torch[0][0][:20])
    print(mel_torch2[0][0][:20])
    print(mel_tf[0][0][:20])


    # === 误差计算 ===
    diff = mel_torch - mel_tf
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))

    print(f"\n🔍 PyTorch vs TensorFlow Mel 对齐结果：")
    print(f"MSE: {mse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"MAX: {max_diff:.6e}")



    # === 误差计算 ===
    diff = mel_torch - mel_torch2
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))

    print(f"\n🔍 PyTorch vs PyTorch2 对齐结果：")
    print(f"MSE: {mse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"MAX: {max_diff:.6e}")