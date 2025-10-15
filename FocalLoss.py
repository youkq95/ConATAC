import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma, weight, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        epsilon = 1e-7
        probs = inputs.softmax(dim=1)
        probs = torch.clamp(probs, min=epsilon, max=1 - epsilon)
        pt = probs[range(len(targets)), targets]
        alpha_t = self.alpha * torch.ones_like(pt)
        alpha_t[targets == 0] = 1 - self.alpha
        if self.weight is not None:
            weight_tensor = self.weight.to(inputs.device)
            class_weights = weight_tensor[targets]
        else:
            class_weights = torch.ones_like(pt)

        log_pt = torch.log(pt)
        loss = -alpha_t * class_weights * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss