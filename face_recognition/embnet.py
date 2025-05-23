import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class EmbNet(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, dim)
    def forward(self, x):
        z = self.backbone(x)
        z = F.normalize(self.fc(z))
        return z