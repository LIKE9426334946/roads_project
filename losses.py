import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        union = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class WeightedCEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 0.5, dice_weight: float = 1.0, class_weights=None):
        super().__init__()
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weight = None
        self.register_buffer("weight_tensor", weight if weight is not None else torch.tensor([]), persistent=False)
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight_tensor if self.weight_tensor.numel() > 0 else None)
        dice = self.dice(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice
