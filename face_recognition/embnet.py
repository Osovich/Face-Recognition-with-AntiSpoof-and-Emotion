import torch.nn.functional as F
import torch, torch.nn as nn
class EmbNet(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(2048)           # “BN-neck” :contentReference[oaicite:4]{index=4}
        self.fc = nn.Linear(2048, dim, bias=False)
        
    def forward(self,x):
        z = self.bn(self.backbone(x))
        z = F.normalize(self.fc(z), dim=1)
        return z