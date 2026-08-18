"""ResNet50 face embedding network, ArcFace head and triplet loss."""

import math

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


class RandomMask:
    """Randomly draw a synthetic surgical mask over an already-aligned face.

    Faces reaching this point are aligned onto the ArcFace template, so the eyes,
    nose and mouth sit at known positions and the mask can be drawn from the
    template itself - no per-image landmark lookup is needed at train time.

    A model trained only on bare faces leans on the nose and mouth, and loses
    most of its signal when they are covered (see the MLFW evaluation). Masking a
    fraction of every epoch forces the embedding to survive on the eyes, brow and
    face outline while keeping one embedding space for masked and unmasked faces.
    """

    COLOURS = [(235, 235, 232), (214, 205, 176), (160, 145, 120), (248, 245, 245),
               (120, 130, 140), (250, 250, 250)]

    def __init__(self, p=0.4, size=SIZE):
        self.p = p
        self.k = size / 112.0            # template is defined at 112 px

    def __call__(self, img):
        import random as _r
        if _r.random() >= self.p:
            return img

        a = np.array(img)
        k = self.k
        # ArcFace template points, scaled to this image size
        rex, rey = 38.2946 * k, 51.6963 * k
        lex, ley = 73.5318 * k, 51.5014 * k
        nx, ny = 56.0252 * k, 71.7366 * k
        rmy = 92.3655 * k
        h, w = a.shape[:2]

        span = lex - rex
        jitter = _r.uniform(-0.06, 0.06) * span
        top = ny - _r.uniform(0.02, 0.22) * span      # how high the mask rides
        chin = min(h - 1, rmy + _r.uniform(0.85, 1.25) * span)
        half = _r.uniform(0.85, 1.05) * span
        cx = (rex + lex) / 2 + jitter

        poly = np.array([
            [cx - half, top + 0.10 * span],
            [cx - half * 1.02, (top + chin) / 2],
            [cx - half * 0.72, chin],
            [cx, chin + 0.10 * span],
            [cx + half * 0.72, chin],
            [cx + half * 1.02, (top + chin) / 2],
            [cx + half, top + 0.10 * span],
            [cx, top - 0.05 * span],
        ], np.int32)

        colour = list(self.COLOURS[_r.randrange(len(self.COLOURS))])
        colour = [max(0, min(255, c + _r.randint(-18, 18))) for c in colour]

        cv2.fillPoly(a, [poly], colour, cv2.LINE_AA)
        for f in (0.35, 0.55, 0.75):                  # pleats
            y = int(top + f * (chin - top))
            cv2.line(a, (int(cx - half * 0.95), y), (int(cx + half * 0.95), y),
                     tuple(int(c * 0.92) for c in colour), 1, cv2.LINE_AA)
        cv2.polylines(a, [poly], True, tuple(int(c * 0.75) for c in colour), 2, cv2.LINE_AA)
        eye_y = (rey + ley) / 2
        for sx, ex in ((cx - half, 0), (cx + half, w - 1)):   # ear straps
            cv2.line(a, (int(sx), int(top + 0.15 * span)), (int(ex), int(eye_y + 0.18 * span)),
                     tuple(int(c * 0.8) for c in colour), 2, cv2.LINE_AA)
        return Image.fromarray(a)


def train_tf(size=SIZE, mask_p=0.0):
    return transforms.Compose([
        transforms.Resize((size, size)),
        RandomMask(p=mask_p, size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.RandomAffine(8, translate=(0.05, 0.05), scale=(0.92, 1.08))], p=0.7),
        transforms.ColorJitter(0.25, 0.25, 0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_tf(size=SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
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
    def __init__(self, dim=512, pretrained=True, dropout=0.4, size=SIZE):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.trunk = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.dim = dim
        self.input_size = size

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
    model = FaceNet(dim=ckpt.get("dim", 512), pretrained=False,
                    size=ckpt.get("size", SIZE))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def embed(model, paths, device, batch_size=128, workers=0):
    """L2-normalised embeddings for paths, with horizontal-flip TTA."""
    tf = eval_tf(getattr(model, "input_size", SIZE))
    loader = DataLoader(PathData(paths, tf), batch_size=batch_size, num_workers=workers,
                        pin_memory=(str(device) == "cuda"))
    model.eval()
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        z = model(x, normalize=False) + model(torch.flip(x, [3]), normalize=False)
        out.append(F.normalize(z).cpu())
    return torch.cat(out).numpy()
