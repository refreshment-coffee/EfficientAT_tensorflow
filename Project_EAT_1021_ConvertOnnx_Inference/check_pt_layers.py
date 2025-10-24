import torch

# 假设权重文件路径
ckpt_path = "./resources/mn10_as_mAP_471.pt"

# 载入权重
state_dict = torch.load(ckpt_path, map_location="cpu")

# 如果是完整checkpoint，需要取出state_dict
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# 输出权重层名字
weight_names = list(state_dict.keys())
print("权重层数量:", len(weight_names))
print("部分权重层名字示例:", weight_names[:10])  # 只打印前10个，避免太长


# 如果你有模型类，可以这样：
# model = MyModel()
# model_layers = [name for name, _ in model.named_modules()]
# print("模型层数量:", len(model_layers))
# print("部分模型层名字:", model_layers[:20])

# 如果没有模型结构，只能对比数量
print("="*50)
print("⚠️ 你没有提供模型结构，暂时只能统计权重层名字。")
