import torch, pathlib
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

_MODELS = pathlib.Path(__file__).parent.parent / "models"
_EMB_CKPT = _MODELS / "embnet_dim256_cos_30k.pth"

train_tf = transforms.Compose([
    transforms.Resize(128), transforms.CenterCrop(112),
    transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)
])

class EmbNet(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, dim)

    def forward(self,x):
        z = self.backbone(x)
        return nn.functional.normalize(self.fc(z), dim=1)

class Embedder:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.net = EmbNet().to(self.device).eval()
        ckpt = torch.load(_EMB_CKPT, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state"])

    @torch.no_grad()
    def embed(self, pil_img):
        t = train_tf(pil_img).unsqueeze(0).to(self.device)
        return self.net(t)[0].cpu()          # 256-D tensor
