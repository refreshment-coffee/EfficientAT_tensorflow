import argparse
import time

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


AUDIO_PATH=r"G:\SYZ_Projects\OriginDatasets\测试_wav\其他情况测试\t_del_1009_cut\0-rain025/0-rain025___0000.WAV"
MODEL_NAME='mn10_as'
PT_NAME="./save_pts/mn10_as_custom22_epoch_20_acc_996.pt"

CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
              '003Bird','004Frog','005Insect',
              '006Dog','007Honk','008Background',
              '009Person','010Other','011Drill',
              '012Knock','013Aerodynamic','014Mech_work',
              '015Chanming','016Cat','017Firecracker','111Chaninsaw','114Engine','018Chiken','019Plane'
              ]
LABELS=CUSTOM_CLASSES
NUM_CLASSES = len(CUSTOM_CLASSES)
CLASS_DIM=NUM_CLASSES
DEVICE= torch.device('cuda')

def Get_model():
    model_name="mn10_as"
    device=DEVICE
    strides=[2, 2, 2, 2]
    head_type="mlp"

    model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                          strides=strides, head_type=head_type, load_pt_path=PT_NAME, num_classes=NUM_CLASSES)
    model.to(device)
    model.eval()
    return model



model=Get_model()
def custom_infer(audio_path):
    # model to preprocess waveform into mel spectrograms
    device=DEVICE
    n_mels=128
    sample_rate=32000
    window_size=800
    hop_size=320
    cuda=True

    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    with torch.no_grad(), autocast(device_type=device.type) if cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    # preds_sig  = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
    #
    # sorted_indexes = np.argsort(preds_sig)[::-1]
    # print("************* Acoustic Event Detected: *****************")
    # for k in range(10):
    #     print('{}: {:.3f}'.format(CUSTOM_CLASSES[sorted_indexes[k]],
    #         preds_sig[sorted_indexes[k]]))

    preds=preds.cpu()
    # print(preds)
    # print(np.array(preds).shape)
    # 计算 softmax 概率
    softmax_probs = torch.softmax(preds, dim=-1).squeeze()  # shape: [num_classes]

    # 获取预测类别和概率
    pred_idx = torch.argmax(softmax_probs).item()
    pred_prob = softmax_probs[pred_idx].item()

    # 打印预测结果

    # Print audio tagging top probabilities


    # print('{}: {:.3f}'.format(CUSTOM_CLASSES[pred_idx],pred_prob))
    # print("********************************************************")


    return pred_idx,pred_prob



import os
all_count_result= [[0 for _ in range(CLASS_DIM)] for _ in range(CLASS_DIM)]
def evaluate_folder(folder_path, expected_label_idx):
    """评估某个文件夹下的所有音频，返回总数和正确数"""
    file_count = 0
    correct_count = 0
    try:
        filenames = os.listdir(folder_path)
    except Exception as e:
        print(f"无法读取目录：{folder_path}\n错误：{e}")
        return 0, 0
    # c=0
    for filename in filenames:
        # c+=1
        # if c>10:
        #     return file_count, correct_count
        if not filename.lower().endswith(".wav"):
            continue
        file_path = os.path.join(folder_path, filename)
        try:

            l, p = custom_infer(file_path)
            # print(l, p)
            # lab,proper,energies_ = infer(wavepath)
            # proper = round(np.float(proper),2)
            # energies_second_sum = round(float(energies_second_sum), 2)
            #
            # proper_1 = round(float(proper_1*100), 2)
            # proper_2 = round(float(proper_2*100), 2)
        # print(res)
            predicted_label = l
            all_count_result[expected_label_idx][predicted_label]+=1

            class_names = LABELS
            # result_dict = {"code": 1, "msg": 'success',
            #                "data": [{"classType": class_names[labs_1], "property": proper_1, "modelType": "1",
            #                          "secondClassType": class_names[labs_2], "secondProperty": proper_2,
            #                          'totalEnergy': str(energies_second_sum)}]}
            # print(result_dict)
            if predicted_label == expected_label_idx:
                    correct_count += 1
            else:
                print(f'audio: {file_path} ---> predict: {LABELS[predicted_label]} (expected: {LABELS[expected_label_idx]})')
            file_count += 1
        except Exception as e:
            print(f"预测出错：{file_path}\n错误：{e}")
    return file_count, correct_count

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Example of parser. ')
    # # model name decides, which pre-trained model is loaded
    # parser.add_argument('--model_name', type=str, default='mn10_as')
    # parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    # parser.add_argument('--head_type', type=str, default="mlp")
    # parser.add_argument('--cuda', action='store_true', default=True)
    # # parser.add_argument('--audio_path', type=str, required=True)
    #
    # # preprocessing
    # parser.add_argument('--sample_rate', type=int, default=32000)
    # parser.add_argument('--window_size', type=int, default=800)
    # parser.add_argument('--hop_size', type=int, default=320)
    # parser.add_argument('--n_mels', type=int, default=128)
    #
    # # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    # parser.add_argument('--ensemble', nargs='+', default=[])
    #
    # args = parser.parse_args()

    # audio_tagging(args)





    start_time=time.time()
    dirs=['fold0/','fold1/','fold2/',
          'fold3/','fold4/','fold5/',
          'fold6/','fold7/','fold8/',
          'fold9/','fold10/','fold11_1/',
          'fold12/','fold13/','fold14/',
          'fold15/','fold16/','fold17/',
    'fold11_2/','fold_engine/','fold18/','fold19/']



    base_dir = r'G:\SYZ_Projects\OriginDatasets\Sounds17Class\test_origin'



    labelindex = list(range(len(LABELS)))
    total_files = 0
    total_correct = 0
    file_counts = [0 for _ in range(CLASS_DIM)]
    correct_counts = [0 for _ in range(CLASS_DIM)]
    accs=[]



    for idx, folder in enumerate(dirs):
        if idx not in [11]:continue
        # if idx not in NEED_IDXS:continue
        folder_path = os.path.join(base_dir, folder)
        fc, cc = evaluate_folder(folder_path, labelindex[idx])
        file_counts[idx]=fc
        correct_counts[idx]=cc
        total_files += fc
        total_correct += cc
        acc = cc / fc if fc > 0 else 0
        accs.append(acc)
        label=LABELS[idx]
        print(f"{label:<8} | total: {fc:4} | correct: {cc:4} | acc: {acc:.2%}")

    print("======================================================================")
    # 输出每一类的准确率
    for idx, label in enumerate(LABELS):
        fc = file_counts[idx]
        cc = correct_counts[idx]
        acc = cc / fc if fc > 0 else 0
        print(f"{label:<8} | total: {fc:4} | correct: {cc:4} | acc: {acc:.2%}")

    # 总体准确率
    overall_acc = total_correct / total_files if total_files > 0 else 0
    print(f"\n Overall Accuracy: {total_correct}/{total_files} = {overall_acc:.2%}")
    print(f" Average Accuracy: {sum(accs)/len(accs):.2%}")

    for t in all_count_result:
        print(t)

    end_time=time.time()
    total_time = end_time - start_time

    # 打印测速结果
    if total_files > 0:
        time_per_sample = total_time / total_files
        samples_per_second = total_files / total_time
        print("\n==================== Speed Test ====================")
        print(f"Total Time Elapsed: {total_time:.2f} seconds")
        print(f"Total Samples     : {total_files}")
        print(f"Time per Sample   : {time_per_sample:.4f} seconds/sample")
        print(f"Samples per Second: {samples_per_second:.2f} samples/sec")

