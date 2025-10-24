import onnxruntime as ort
import numpy as np
import os


import numpy as np
import resampy
from scipy.io import wavfile

import argparse
import time
from random import random


import librosa
import numpy as np


from models.preprocess import AugmentMelSTFTTF





# CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
#               '003Bird','004Frog','005Insect',
#               '006Dog','007Honk','008Background',
#               '009Person','010Other','011Drill',
#               '012Knock','013Aerodynamic','014Mech_work',
#               '015Chanming','016Cat','017Firecracker','018Chiken','019Plane'
#               ]
LABELS = [
        'rain', 'wind', 'thunder', 'bird', 'frog', 'insects', 'dog', 'honk', 'other',
        'person', 'other', 'chainsaw', 'knock', 'aerodynamic', 'engine', 'cicada', 'cat',

    'firecracker','chicken','airplane'
    ]
# LABELS=labels
NUM_CLASSES = len(LABELS)
CLASS_DIM=NUM_CLASSES

ONNX_NAME='./save_onnxs/EAT_dymn20_as_128_200_1021_03_986.onnx'


def calculate_energy(signal):
    return np.sum(np.abs(signal) ** 2)



def Get_model():  ###通过onnx来推理


    # 加载模型
    model = ort.InferenceSession(ONNX_NAME)

    # model.eval()
    return model


ort_session=Get_model()
output_names = [o.name for o in ort_session.get_outputs()]
def custom_infer(audio_path):
    # model to preprocess waveform into mel spectrograms
    n_mels=128
    sample_rate=32000
    window_size=800
    hop_size=320
    target_sec=2
    cuda=True

    mel = AugmentMelSTFTTF(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size,fmax=15000)

    # mel.to(device)
    # mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)


    ###固定长度 2s

    target_len = int(sample_rate * target_sec)
    cur_len = len(waveform)

    # 过短 补长
    if cur_len < target_len:
        pad_len = target_len - cur_len
        # 在尾部补零
        waveform = np.pad(waveform, (0, pad_len), mode="constant")

    # 过长  裁剪
    elif cur_len > target_len:

            # 随机裁剪
        # start = random.randint(0, cur_len - target_len)

            # 中心裁剪（推理时稳定）
        start = (cur_len - target_len) // 2
        waveform = waveform[start:start + target_len]
    # waveform = torch.from_numpy(waveform[None, :]).to(device)
    energy_sum=calculate_energy(waveform)

    waveform=np.expand_dims(waveform,axis=0)
    # return waveform.astype(np.float32)




    # print(waveform[:10])

    # print(waveform.shape)
    # with torch.no_grad(), autocast(device_type=device.type) if cuda else nullcontext():
    spec = mel(waveform)
    # print(spec.shape)  #(1,200, 128)
    # spec=np.transpose(spec, (0, 2, 1))  ##1,128,200
    # print(spec.shape) ##(1, 1, 128, 200)
    new_inputs=np.expand_dims(spec,axis=0)


    # new_inputs = new_inputs.detach().cpu().numpy().astype(np.float32) ## torch--->numpy
        # print(spec.shape)  ###1,128,200
    outputs = ort_session.run(output_names, {"input_values": new_inputs})

    preds = outputs[0]
    # preds=preds[0] ##bs=1
    # features = outputs[1]
    # print(preds)
    # print(features)
    # exit()

    preds=preds
    # print(preds)
    # print(np.array(preds).shape)
    # 计算 softmax 概率
    preds = np.squeeze(preds)  # [num_classes]

    # softmax
    exp_preds = np.exp(preds - np.max(preds))  # 避免溢出
    softmax_probs = exp_preds / np.sum(exp_preds)

    # # 获取预测类别和概率
    # pred_idx = int(np.argmax(softmax_probs))
    # pred_prob = float(softmax_probs[pred_idx])
    # pred_idx, pred_prob

    lab1,lab2 = np.argsort(softmax_probs)[-2:][::-1]

    # 获取对应概率
    prob1=softmax_probs[lab1]
    prob2=softmax_probs[lab2]
    # top2_probs = softmax_probs[top2_idx]

    # print("前两个最大类别索引: {} {} ".format( lab1,lab2))
    # print("对应概率: {} {} ".format( prob1,prob2))

    # 打印预测结果

    # Print audio tagging top probabilities


    # print('{}: {:.3f}'.format(CUSTOM_CLASSES[pred_idx],pred_prob))
    # print("********************************************************")

    # exit()
    return lab1,prob1,lab2,prob2,energy_sum

    ##energies_sum = float(sum(energies))
    ##labs_1, proper_1, labs_2, proper_2, energies_second_sum

import wave

from flask import Flask, request
from flask_cors import CORS
import json
import base64
import tensorflow as tf

import numpy as np
import resampy

# from models.ModelInProcessor import runFirst, getProcessor  ## 内存泄漏管理

processor1 = None
processor2 = None

"""
EAT_flask
"""

app = Flask(__name__)
CORS(app)


SR=32000


def fix_wav_header(input_path):
    output_path=input_path
    try:
        with wave.open(input_path, 'rb') as wf_in:
            params = wf_in.getparams()
            frames = wf_in.readframes(params.nframes)
    except wave.Error as e:
        print(f"Error reading input WAV: {e}")
        return False

    try:
        with wave.open(output_path, 'wb') as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(frames)
    except wave.Error as e:
        print(f"Error writing fixed WAV: {e}")
        return False

    return True



def custom_infer(audio_path):
    # model to preprocess waveform into mel spectrograms
    n_mels=128
    sample_rate=32000
    window_size=800
    hop_size=320
    target_sec=2
    cuda=True

    mel = AugmentMelSTFTTF(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size,fmax=15000)

    # mel.to(device)
    # mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)


    ###固定长度 2s

    target_len = int(sample_rate * target_sec)
    cur_len = len(waveform)

    # 过短 补长
    if cur_len < target_len:
        pad_len = target_len - cur_len
        # 在尾部补零
        waveform = np.pad(waveform, (0, pad_len), mode="constant")

    # 过长  裁剪
    elif cur_len > target_len:

            # 随机裁剪
        # start = random.randint(0, cur_len - target_len)

            # 中心裁剪（推理时稳定）
        start = (cur_len - target_len) // 2
        waveform = waveform[start:start + target_len]
    # waveform = torch.from_numpy(waveform[None, :]).to(device)
    energy_sum=calculate_energy(waveform)

    waveform=np.expand_dims(waveform,axis=0)
    # return waveform.astype(np.float32)




    # print(waveform[:10])

    # print(waveform.shape)
    # with torch.no_grad(), autocast(device_type=device.type) if cuda else nullcontext():
    spec = mel(waveform)
    # print(spec.shape)  #(1,200, 128)
    # spec=np.transpose(spec, (0, 2, 1))  ##1,128,200
    # print(spec.shape) ##(1, 1, 128, 200)
    new_inputs=np.expand_dims(spec,axis=0)


    # new_inputs = new_inputs.detach().cpu().numpy().astype(np.float32) ## torch--->numpy
        # print(spec.shape)  ###1,128,200
    outputs = ort_session.run(output_names, {"input_values": new_inputs})

    preds = outputs[0]
    # preds=preds[0] ##bs=1
    # features = outputs[1]
    # print(preds)
    # print(features)
    # exit()

    preds=preds
    # print(preds)
    # print(np.array(preds).shape)
    # 计算 softmax 概率
    preds = np.squeeze(preds)  # [num_classes]

    # softmax
    exp_preds = np.exp(preds - np.max(preds))  # 避免溢出
    softmax_probs = exp_preds / np.sum(exp_preds)

    # # 获取预测类别和概率
    # pred_idx = int(np.argmax(softmax_probs))
    # pred_prob = float(softmax_probs[pred_idx])
    # pred_idx, pred_prob

    lab1,lab2 = np.argsort(softmax_probs)[-2:][::-1]

    # 获取对应概率
    prob1=softmax_probs[lab1]
    prob2=softmax_probs[lab2]
    # top2_probs = softmax_probs[top2_idx]

    # print("前两个最大类别索引: {} {} ".format( lab1,lab2))
    # print("对应概率: {} {} ".format( prob1,prob2))

    # 打印预测结果

    # Print audio tagging top probabilities


    # print('{}: {:.3f}'.format(CUSTOM_CLASSES[pred_idx],pred_prob))
    # print("********************************************************")

    # exit()
    return lab1,prob1,lab2,prob2,energy_sum

    ##energies_sum = float(sum(energies))
    ##labs_1, proper_1, labs_2, proper_2, energies_second_sum



# import psutil
# import gc

@app.route('/api/getBase64FilePredict', methods=['POST'])
def process_data():
    data = json.loads(request.get_data())
    # encode_wav = base64.b64encode(open("./audio/fold0-rain/0-rain919.wav", "rb").read())
    wavepath = "./audio_temp/" + 'temp.wav'
    wav_file = open(wavepath, "wb")
    decode_wav = base64.b64decode(data['base64Data'])
    wav_file.write(decode_wav)

    try:
        sr, wav_data = wavfile.read(wavepath)

    except Exception as e:
        # print("wav_fail")
        success=fix_wav_header(wavepath)
        if not success:
            print("\nI0IError \n:", e)
            result_dict = {"code": 0, "msg": 'fail', "data": []}
            return result_dict


    EEerror= {"code": "0", "msg": "fail", "data": []}
    result_dict = {"code":1, "msg":'success', "data": [{"classType":None, "property": "98.00", "modelType":"1"}]}
    try:
        # print("Before infer Memory (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        # print(f"[INFO] 当前进程数: {len(psutil.Process().children(recursive=True))}")
        # for child in  psutil.Process().children(recursive=True):
        #     print(f"[flask代码]子进程 PID: {child.pid}")
        labs_1, proper_1, labs_2, proper_2, energies_second_sum = custom_infer(wavepath)

        # global PLANE_COUNT
        # PLANE_COUNT = max(0, PLANE_COUNT - 1)
        energies_second_sum="{:.2f}".format(energies_second_sum * 100)
        proper_1="{:.2f}".format(proper_1 * 100)
        proper_2 = "{:.2f}".format(proper_2 * 100)

        if proper_1 == 'nan' or proper_2 == 'nan':
            proper_1 = '98.80'
            proper_2 = '1.00'
        else:
            proper_1 = proper_1
            proper_2 = proper_2
        class_names = LABELS

        result_dict = {"code":1, "msg":'success', "data": [{"classType":class_names[labs_1],"property":proper_1, "modelType":"1",
                                                            "secondClassType":class_names[labs_2],"secondProperty":proper_2, 'totalEnergy':"["+str(energies_second_sum)+"]"}]}
    except Exception as e:
        print("\nI0IError \n:", e)
        return EEerror
    return json.dumps(result_dict)

    # return {"code": "1", "msg": "success", "data": [{"classType": "other", "property": "98.00", "modelType": "1"}]}

@app.route('/api/getRunStatus', methods=['POST'])
def module_state():
    # EEerror= {"code": "0", "msg": "fail", "data": []}
    try:

        ort_session = Get_model()
        output_names = [o.name for o in ort_session.get_outputs()]

        result_dict = {"code":1, "msg":'success',"data": [{"statusType": "0"}]}
        # if model.load and infer:
        #     result_dict = {"code":1, "msg":'success',"data": [{"statusType": "0"}]}
        # else:
        #     result_dict = {"code":0, "msg":'fail',"data": []}
    except Exception as e:
        print("\nI0IError \n:", e)
        result_dict = {"code":0, "msg":'fail',"data": []}
        return result_dict
    return json.dumps(result_dict)


if __name__ == "__main__":
    # runFirst()
    app.run(host='0.0.0.0',port=6109, debug=True)








