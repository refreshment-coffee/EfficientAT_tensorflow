
import torch.nn as nn


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
        if len(features.shape) == 3:
            features = features.mean(axis=1)
        # print(features.shape)
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  ##同label

        # print("features.shape ",features.shape)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)  # 避免自身对比
        mask = mask * logits_mask  ##非自身的同label

        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

