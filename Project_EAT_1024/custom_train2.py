from collections import Counter

import torch
import torch.nn.functional as F
from utils_syz import ClassBalancedLoss, SupConLoss

import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets','helpers')))
from custom_temp import get_custom22_valid_set, get_custom22_train_set,Custom22Dataset

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup


# --------------- 自定义 22 类 ----------------
CUSTOM_CLASSES =['000Rain','001Wind','002Thunder',
              '003Bird','004Frog','005Insect',
              '006Dog','007Honk','008Background',
              '009Person','010Other','011Drill',
              '012Knock','013Aerodynamic','014Mech_work',
              '015Chanming','016Cat','017Firecracker','111Chaninsaw','114Engine','018Chiken','019Plane'
              ]
NUM_CLASSES = len(CUSTOM_CLASSES)


import wandb

wandb.login(key="b5295f2bad5bc3d48c88ce70cee5a11ca2025594")  # 替换为你自己的 key

def compute_samples_per_cls(dataset):
    label_counter = Counter()
    for _,_, label in tqdm(dataset):
        if torch.is_tensor(label):
            label = label
        label_counter[label] += 1

    # 将结果按类别顺序组织成列表
    max_label = max(label_counter)
    samples_per_cls = [label_counter[i] for i in range(max_label + 1)]
    return samples_per_cls

def train(args):
    wandb.init(
        project="Custom22",
        notes="Fine-tune Models on Custom22 dataset.",
        tags=["Audio Classification", "Fine-Tuning"],
        config=args,
        name=args.experiment_name
    )

    device = torch.device('cuda')

    # Mel-Spectrogram
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         freqm=args.freqm,
                         timem=args.timem,
                         fmin=args.fmin,
                         fmax=args.fmax,
                         fmin_aug_range=args.fmin_aug_range,
                         fmax_aug_range=args.fmax_aug_range)
    mel.to(device)

    # 模型
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    print("width: ",width)
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=args.pretrain_final_temp,
                         num_classes=NUM_CLASSES)
    else:
        model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=NUM_CLASSES)
    model.to(device)

    # 训练 DataLoader
    dl = DataLoader(dataset=get_custom22_train_set(resample_rate=args.resample_rate,
                                                   roll=False if args.no_roll else True,
                                                   wavmix=False if args.no_wavmix else True,
                                                   gain_augment=args.gain_augment),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

    # 验证 DataLoader
    eval_dl = DataLoader(dataset=get_custom22_valid_set(resample_rate=args.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)

    ##查看model的层
    # model_layers = [name for name, _ in model.named_modules()]
    # print("模型层数量:", len(model_layers))
    # print("部分模型层名字:", model_layers)
    # exit()
    # finetune_keywords = [
    #     "classifier",  # 分类头
    #     "features.13", "features.14",
    #     "features.15", "features.16",
    # ]
#     finetune_keywords = [
#          "layers.9", 
#             "layers.9",   
#         "layers.10", 
#             "layers.11", 
#     "layers.12", 
#     "layers.13", 
#     "layers.14", 
#     "in_c",
#     "out_c", 
#     "classifier"
# ]

#     # 冻结参数：默认全部冻结
#     for name, param in model.named_parameters():
#         param.requires_grad = False

#     # 只解冻最后几层和分类头
#     for name, param in model.named_parameters():
#         if any(key in name for key in finetune_keywords):
#             param.requires_grad = True
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"模型总参数量: {total/1e6:.2f}M, 可训练参数量: {trainable/1e6:.2f}M")

    samples_per_cls=[14562, 9007, 8266, 15027, 6334, 11761, 5067, 3827, 9314, 7505, 5938, 3354, 4332, 3196, 1607, 5127, 2942, 3482, 3784, 2917, 3019, 9261]
    # samples_per_cls = [14562, 9007, 8266, 15027, 6334, 11761, 5067, 3827, 9314, 7505, 5938, 3354, 4332, 3196, 1607,
    #                    5127, 2942, 3482, 3784, 2917, 3019, 9261]
    if samples_per_cls == None:
        samples_per_cls_temp_dataset =DataLoader(dataset=get_custom22_train_set(resample_rate=args.resample_rate,
                                                   roll=False if args.no_roll else True,
                                                   wavmix=False if args.no_wavmix else True,
                                                   gain_augment=args.gain_augment),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

        samples_per_cls = compute_samples_per_cls(samples_per_cls_temp_dataset)
        print("samples_per_cls: ",samples_per_cls)

    CBLoss = ClassBalancedLoss(beta=0.9999, samples_per_cls=samples_per_cls, num_classes=NUM_CLASSES)
    SCLoss = SupConLoss(temperature=0.07)  # temperature 可调整

    # 优化器：只优化 requires_grad=True 的参数（尊重你之前冻结的设置）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # scheduler 保持你原来的 schedule_lambda / LambdaLR 方式
    schedule_lambda = exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    # 超参：loss 权重（可改）
    cb_w =  0.8          # CBLoss 权重
    sup_w =  0.2   # SupCon 权重
   
    name = None
    accuracy, val_loss = float('NaN'), float('NaN')
    for epoch in range(args.n_epochs):
        mel.train()
        model.train()

        train_stats = {"train_loss": [], "cb_loss": [], "supcon_loss": [], "ce_loss": []}
        pbar = tqdm(dl)
        pbar.set_description(f"Epoch {epoch+1}/{args.n_epochs}: accuracy: {accuracy:.4f}, val_loss: {val_loss:.4f}")
        for batch in pbar:
            x, f, y = batch                
            bs = x.size(0)
            x = x.to(device)
            # print("label:  ",y)
            # ensure y is long tensor for CE, and also on device
            # if isinstance(y, torch.Tensor):
            y = y.to(device)
            # else:
                # y = torch.tensor(y, dtype=torch.long, device=device)

            # 前置 zero_grad
            optimizer.zero_grad()

            # forward: mel + model
            x_mel = _mel_forward(x, mel)     # shape e.g. [B, C, H, W]  (你原来已实现)
            # 模型返回 (logits, embedding) —— 保持与原接口一致
            y_hat, embedding = model(x_mel)  

            cb_loss_val = CBLoss(y_hat, y) 

            # --- 2) SupConLoss: contrastive on embeddings
            # SupConLoss 通常接受形状 [B, feat_dim]，并且 labels (long)
            # 可能你的 embedding 形状不同（例如 [B, 1, feat_dim]），做兼容性处理：
            emb = embedding
            # print("emb.dim: ",emb.dim())
            # if emb.dim() == 3 and emb.size(1) == 1:
            #     emb = emb.squeeze(1)   # make [B, feat_dim]
            # # 如果是多时序向量，如 [B, T, D]，可以先做池化（mean）或取 CLS token:
            # if emb.dim() == 3:  # [B, T, D]
            #     emb = emb.mean(dim=1)
            supcon_loss_val = SCLoss(emb, y)

            # --- 3) CrossEntropyLoss: standard CE
            # ce_loss_val = F.cross_entropy(y_hat, y)

            # --- 合并 loss
            loss = cb_w * cb_loss_val + sup_w * supcon_loss_val 
            # print("loss: ",np.array(loss.cpu().detach().numpy()).shape)
            # backward + step
            loss = loss.mean()
            train_stats['train_loss'].append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        # 验证
        accuracy, val_loss = _test(model, mel, eval_dl, device)

        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "accuracy": accuracy,
                   "val_loss": val_loss})

        # 保存模型
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = f"{model_name}_custom22_epoch_{epoch}_acc_{int(round(accuracy * 1000))}.pt"
        print("model_save: ", os.path.join(wandb.run.dir, name))
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))
        if epoch and epoch %5==0:
            torch.save(model.state_dict(), os.path.join("./save_temp_pts", name))

def _mel_forward(x, mel):
    old_shape = x.size()    #(128, 1, 64000)
    x = x.reshape(-1, old_shape[2])  
    # print(x.shape)##torch.Size([128, 64000])   ##bs,64000
    x = mel(x)
    # print(x.shape)  #torch.Size([128, 128, 200])
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    # print(x.shape)  #torch.Size([128, 1, 128, 200])
    # exit()
    # print(x)
    return x


def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()
    targets, outputs, losses = [], [], []

    
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, f, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    accuracy = metrics.accuracy_score(targets, outputs.argmax(axis=1))

    # exit()
    return accuracy, losses.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom22 Audio Classification Training')
    parser.add_argument('--experiment_name', type=str, default="Custom22")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fold', type=int, default=1)

    # training
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--model_name', type=str, default="dymn20_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=2)  ##default=12
    parser.add_argument('--weight_decay', type=int, default=0.0)




    # lr schedule
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--warm_up_len', type=int, default=10)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=65)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()

    device = torch.device('cuda')

    print(f"Using device: {device}")

    train(args)
