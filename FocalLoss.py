import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma, weight, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用softmax后的概率作为pt
        epsilon = 1e-7
        probs = inputs.softmax(dim=1)
        # 防止极端值
        probs = torch.clamp(probs, min=epsilon, max=1 - epsilon)
        pt = probs[range(len(targets)), targets]

        # 计算 alpha_t
        # alpha和gamma一起控制损失的权重。其中gamma控制的是难易程度，alpha控制的是数量比例（基本上）
        alpha_t = self.alpha * torch.ones_like(pt)
        alpha_t[targets == 0] = 1 - self.alpha

        # 处理类别权重
        if self.weight is not None:
            # 确保权重在同一设备上
            weight_tensor = self.weight.to(inputs.device)
            # 获取每个目标类别对应的权重
            class_weights = weight_tensor[targets]
        else:
            class_weights = torch.ones_like(pt)

        # 计算 Focal Loss
        log_pt = torch.log(pt)
        loss = -alpha_t * class_weights * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss