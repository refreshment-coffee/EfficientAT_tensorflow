import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels


AUDIO_PATH="./test_audios/0-rain025.WAV"
MODEL_NAME='mn10_as'
PT_NAME="./save_pts/mn10_as_custom22_epoch_6_acc_972.pt"

CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
              '003Bird','004Frog','005Insect',
              '006Dog','007Honk','008Background',
              '009Person','010Other','011Drill',
              '012Knock','013Aerodynamic','014Mech_work',
              '015Chanming','016Cat','017Firecracker','111Chaninsaw','114Engine','018Chiken','019Plane'
              ]
NUM_CLASSES = len(CUSTOM_CLASSES)




def audio_tagging(args):
    """
    Running Inference on an audio clip.
    """
    model_name = MODEL_NAME
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path =AUDIO_PATH
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides, head_type=args.head_type,load_pt_path=PT_NAME,num_classes=NUM_CLASSES)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds_sig = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds_sig)[::-1]
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print('{}: {:.3f}'.format(CUSTOM_CLASSES[sorted_indexes[k]],
            preds_sig[sorted_indexes[k]]))

    preds=preds.cpu()
    print(preds)
    # print(np.array(preds).shape)
    # 计算 softmax 概率
    softmax_probs = torch.softmax(preds, dim=-1).squeeze()  # shape: [num_classes]

    # 获取预测类别和概率
    pred_idx = torch.argmax(softmax_probs).item()
    pred_prob = softmax_probs[pred_idx].item()

    # 打印预测结果

    # Print audio tagging top probabilities


    print('{}: {:.3f}'.format(CUSTOM_CLASSES[pred_idx],pred_prob))
    print("********************************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn10_as')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=True)
    # parser.add_argument('--audio_path', type=str, required=True)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()

    audio_tagging(args)
