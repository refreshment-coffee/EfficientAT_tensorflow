import torch

# 本地加载模型


PT_NAME="./save_pts/dymn20_as_1021_custom22_epoch_25_acc_1000.pt"
save_onnx_name="../save_onnxs/EAT_dymn20_as_128_200_1022_25_1000_best.onnx"

##拆fold11   拆fold14
# CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
#               '003Bird','004Frog','005Insect',
#               '006Dog','007Honk','008Background',
#               '009Person','010Other','011Drill',
#               '012Knock','013Aerodynamic','014Mech_work',
#               '015Chanming','016Cat','017Firecracker','111Chaninsaw','114Engine','018Chiken','019Plane'
#               ]


CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
              '003Bird','004Frog','005Insect',
              '006Dog','007Honk','008Background',
              '009Person','010Other','011Drill',
              '012Knock','013Aerodynamic','014Mech_work',
              '015Chanming','016Cat','017Firecracker','018Chiken','019Plane'
              ]
LABELS=CUSTOM_CLASSES
NUM_CLASSES = len(CUSTOM_CLASSES)
CLASS_DIM=NUM_CLASSES


DEVICE= torch.device('cuda')
from models.dymn.model import get_model as get_dymn

def Get_model():
    model_name="dymn20_as"
    device=DEVICE
    strides=[2, 2, 2, 2]
    head_type="mlp"
    width=2.0   ### dymn20_as---width=2.0
    pretrain_final_temp=1.0
    model = get_dymn(width_mult=width, pretrained_name=model_name,
                         pretrain_final_temp=pretrain_final_temp,
                         num_classes=NUM_CLASSES,load_pt_path=PT_NAME)

    model.to(device)
    model.eval()
    return model


model =Get_model()


# 创建 dummy 输入（和真实的 input_values shape 一致）
dummy_input = torch.randn(1,1,128,200)  # [batch, mel_bins, time_steps]
# dummy_input=dummy_input.unsqueeze(0)
# print(dummy_input.shape)
dummy_input=dummy_input.to(DEVICE)


# 导出为 ONNX 文件
torch.onnx.export(
    model,
    dummy_input,
    save_onnx_name,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch_size", 2: "time_steps"},
        "logits": {0: "batch_size"}
    }
)

### 对应onnx数据获取情况：
# ort_session=Get_model()
# output_names = [o.name for o in ort_session.get_outputs()]
# outputs = ort_session.run(output_names, {"input_values": new_inputs}) ##new_inputs最好是Numpy形式。
#
# preds = outputs[0]
# print(preds)



# torch.onnx.export(
#     model,
#     dummy_input,
#     save_onnx_name,
#     export_params=True,
#     opset_version=13,
#     do_constant_folding=True,
#     input_names=["input_values"],
#     output_names=["logits", "features"],  # ✅ 添加第二个输出
#     dynamic_axes={
#         "input_values": {0: "batch_size", 2: "time_steps"},
#         "logits": {0: "batch_size"},
#         "features": {0: "batch_size"}
#     }
# )
### 对应onnx数据获取情况：
# ort_session=Get_model()
# output_names = [o.name for o in ort_session.get_outputs()]
# outputs = ort_session.run(output_names, {"input_values": new_inputs}) ##new_inputs最好是Numpy形式。
#
# preds = outputs[0]
# features = outputs[1]
# print(preds)
# print(features)


print(f" 导出完成：{save_onnx_name}")
