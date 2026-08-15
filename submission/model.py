"""ResNet50 face embedding network, ArcFace head and triplet loss."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 112


def train_tf():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.RandomAffine(8, translate=(0.05, 0.05), scale=(0.92, 1.08))], p=0.7),
        transforms.ColorJitter(0.25, 0.25, 0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_tf():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class PathData(Dataset):
    def __init__(self, paths, tf=None):
        self.paths = list(paths)
        self.tf = tf or eval_tf()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class FaceNet(nn.Module):
    def __init__(self, dim=512, pretrained=True, dropout=0.4):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.trunk = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.dim = dim

    def forward(self, x, normalize=True):
        z = self.head(self.trunk(x).flatten(1))
        return F.normalize(z) if normalize else z


class ArcFace(nn.Module):
    def __init__(self, dim, n_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.xavier_normal_(self.w)
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, z, y):
        cos = F.linear(F.normalize(z), F.normalize(self.w)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1 - cos ** 2)
        phi = cos * self.cos_m - sin * self.sin_m
        # past pi - m the additive margin stops being monotonic
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = torch.zeros_like(cos).scatter_(1, y.view(-1, 1), 1.0)
        return torch.where(onehot.bool(), phi, cos) * self.scale


def triplet_loss(z, y, margin=0.3):
    d = 1.0 - z @ z.t()
    same = y.view(-1, 1).eq(y.view(1, -1))
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos, neg = same & ~eye, ~same

    ok = pos.any(1) & neg.any(1)
    if not ok.any():
        return z.new_zeros(())

    hard_pos = (d - (~pos).float() * 1e4).max(1).values
    hard_neg = (d + (~neg).float() * 1e4).min(1).values
    return F.relu(hard_pos[ok] - hard_neg[ok] + margin).mean()


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FaceNet(dim=ckpt.get("dim", 512), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def embed(model, paths, device, batch_size=128, workers=0):
    """L2-normalised embeddings for paths, with horizontal-flip TTA."""
    loader = DataLoader(PathData(paths), batch_size=batch_size, num_workers=workers,
                        pin_memory=(str(device) == "cuda"))
    model.eval()
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        z = model(x, normalize=False) + model(torch.flip(x, [3]), normalize=False)
        out.append(F.normalize(z).cpu())
    return torch.cat(out).numpy()
