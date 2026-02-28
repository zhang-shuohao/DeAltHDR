import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
# from mmagic.registry import MODELS
from typing import Optional

class L1BaseLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1BaseLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        l1_loss = nn.L1Loss()
        # print('tar:{}'.format(target.shape))
        # print('pred:{}'.format(pred.shape))
        l1_base = l1_loss(pred, target)
        return l1_base

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class L1RankLoss(torch.nn.Module):


    def __init__(self):
        super(L1RankLoss, self).__init__()
        self.l1_w = 0.5
        self.rank_w = 1
        self.hard_thred = 1
        self.use_margin = False
        self.batchsize = 8

    def forward(self, preds, gts):
        b, c, h, w = preds.size()  # 获取形状 b, c, h, w

        # 展开 preds 和 gts 为 [b, c*h*w]
        preds = preds.view(b, -1)  # preds 形状变为 [b, c*h*w]
        gts = gts.view(b, -1)      # gts 形状变为 [b, c*h*w]

        # 计算 L1 损失
        # l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # 排名损失部分
        # 扩展 preds 和 gts，以便进行每对样本的比较
        preds_expanded = preds.unsqueeze(1).repeat(1, b, 1)  # [b, b, c*h*w]
        gts_expanded = gts.unsqueeze(0).repeat(b, 1, 1)      # [b, b, c*h*w]

        # 计算排名损失的 mask
        # 根据每对样本的目标之间的差异计算标志，sign(差异)
        masks = torch.sign(gts_expanded - gts_expanded.transpose(0, 1))  # [b, b, c*h*w]
        masks_hard = (torch.abs(gts_expanded - gts_expanded.transpose(0, 1)) < self.hard_thred) & \
                     (torch.abs(gts_expanded - gts_expanded.transpose(0, 1)) > 0)

        # 计算 em,n 目标：当 preds 确定排名时，计算阈值项 e_{m,n}
        em_n = torch.where(preds_expanded >= preds_expanded.transpose(0, 1),
                           gts_expanded - gts_expanded.transpose(0, 1), 
                           gts_expanded.transpose(0, 1) - gts_expanded)

        # 计算每对视频的排名损失 Lm,n
        rank_loss = torch.relu(torch.abs(preds_expanded - preds_expanded.transpose(0, 1)) - em_n)
        
        # 根据 mask 限制有效的损失
        rank_loss = masks_hard * rank_loss
        
        # 求和并规约
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-8)  # 防止除零错误

        # 总损失
        # loss_total = l1_loss + rank_loss * self.rank_w
        # return loss_total
        return rank_loss

import torch
import torchvision
from torch.cuda.amp import autocast

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # 定义 VGG16 不同特征层
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())

        # 冻结 VGG16 的参数
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        
        # 将特征块保存为 ModuleList
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

        # 注册 mean 和 std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.resize = resize

        # 将模型权重迁移到 GPU
        for block in self.blocks:
            block.to(device)

    def forward(self, input, target, swd=False, feature_layers=[0, 1, 2, 3], style_layers=[]):
        # 如果输入是灰度图，转为 3 通道
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 动态将 mean 和 std 移动到输入的设备上
        mean = self.mean.to(input.device, dtype=input.dtype)
        std = self.std.to(input.device, dtype=input.dtype)

        # 强制将输入和目标转换为 FP32
        input = input.float()
        target = target.float()

        # 标准化
        input = (input - mean) / std
        target = (target - mean) / std

        # 如果需要调整输入大小
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        # 初始化损失
        loss = 0.0
        x = input
        y = target

        with autocast():  # 自动处理混合精度
            # 遍历每个 VGG 特征块
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)

                # 特征损失
                if i in feature_layers:
                    loss += torch.nn.functional.l1_loss(x, y)

                # 样式损失（Gram 矩阵）
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return loss

# class CharbonnierLoss(nn.Module):
#     """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant
#     of L1Loss).

#     Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
#         Super-Resolution".

#     Args:
#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#         sample_wise (bool): Whether calculate the loss sample-wise. This
#             argument only takes effect when `reduction` is 'mean' and `weight`
#             (argument of `forward()`) is not None. It will first reduces loss
#             with 'mean' per-sample, and then it means over all the samples.
#             Default: False.
#         eps (float): A value used to control the curvature near zero.
#             Default: 1e-12.
#     """

#     def __init__(self,
#                  loss_weight: float = 1.0,
#                  reduction: str = 'mean',
#                  sample_wise: bool = False,
#                  eps: float = 1e-12) -> None:
#         super().__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.sample_wise = sample_wise
#         self.eps = eps

#     def forward(self,
#                 pred: torch.Tensor,
#                 target: torch.Tensor,
#                 weight: Optional[torch.Tensor] = None,
#                 **kwargs) -> torch.Tensor:
#         """Forward Function.

#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         return self.loss_weight * charbonnier_loss(
#             pred,
#             target,
#             weight,
#             eps=self.eps,
#             reduction=self.reduction,
#             sample_wise=self.sample_wise)
