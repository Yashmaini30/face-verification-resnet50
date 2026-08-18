"""Fine-tune ResNet50 with ArcFace + batch-hard triplet on the face dataset."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from dataset_preparation import load_split
from generate_pairs import make_pairs
from model import ArcFace, FaceNet, embed, train_tf, triplet_loss
from roc_analysis import metrics


class FaceData(Dataset):
    def __init__(self, root, split, tf):
        people = load_split(root, split)
        self.classes = sorted(people)
        self.tf = tf
        self.samples = []
        self.by_class = {}
        for i, name in enumerate(self.classes):
            paths = people[name]
            self.by_class[i] = list(range(len(self.samples), len(self.samples) + len(paths)))
            self.samples += [(p, i) for p in paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), label


class PKSampler(Sampler):
    """P identities x K images per batch, so triplet mining always has positives."""

    def __init__(self, dataset, p=16, k=4, batches=None):
        self.by_class = {c: idx for c, idx in dataset.by_class.items() if idx}
        self.p = min(p, len(self.by_class))
        self.k = k
        self.batches = batches or max(1, len(dataset) // (self.p * self.k))
        self.rng = random.Random(0)

    def __len__(self):
        return self.batches

    def __iter__(self):
        classes = list(self.by_class)
        for _ in range(self.batches):
            batch = []
            for c in self.rng.sample(classes, self.p):
                pool = self.by_class[c]
                if len(pool) >= self.k:
                    batch += self.rng.sample(pool, self.k)
                else:
                    batch += [self.rng.choice(pool) for _ in range(self.k)]
            yield batch


@torch.no_grad()
def validate(model, root, device, n_pairs, seed, workers):
    people = load_split(root, "val")
    pairs, _ = make_pairs(people, n_pairs // 2, n_pairs // 2, seed)

    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    idx = {p: i for i, p in enumerate(paths)}
    z = embed(model, paths, device, workers=workers)

    a = np.array([idx[p] for p, _, _ in pairs])
    b = np.array([idx[p] for _, p, _ in pairs])
    y = np.array([t for _, _, t in pairs])

    m = metrics(np.sum(z[a] * z[b], axis=1), y)
    m.pop("roc_curve")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--mask-p", type=float, default=0.0,
                    help="fraction of training images given a synthetic mask")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-p", type=int, default=16)
    ap.add_argument("--batch-k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--arc-scale", type=float, default=30.0)
    ap.add_argument("--arc-margin", type=float, default=0.5)
    ap.add_argument("--triplet-weight", type=float, default=1.0)
    ap.add_argument("--triplet-margin", type=float, default=0.3)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batches-per-epoch", type=int, default=None)
    ap.add_argument("--val-pairs", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  mask augmentation p={args.mask_p}")

    ds = FaceData(args.root, "train", train_tf(args.size, args.mask_p))
    sampler = PKSampler(ds, args.batch_p, args.batch_k, args.batches_per_epoch)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.workers > 0)
    print(f"{len(ds)} images / {len(ds.classes)} identities, {len(sampler)} batches "
          f"of {args.batch_p * args.batch_k}")

    model = FaceNet(args.dim, size=args.size).to(device)
    head = ArcFace(args.dim, len(ds.classes), args.arc_scale, args.arc_margin).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    opt = torch.optim.AdamW([
        {"params": model.trunk.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * args.head_lr_mult},
        {"params": head.parameters(), "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.weight_decay)

    steps = len(sampler)
    warmup = args.warmup_epochs * steps
    total = args.epochs * steps

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        t0 = time.time()
        tot = {"loss": 0.0, "arc": 0.0, "tri": 0.0, "hit": 0.0, "n": 0.0}

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                z = model(x)
                logits = head(z, y)
                arc = criterion(logits, y)
                tri = triplet_loss(z.float(), y, args.triplet_margin)
                loss = arc + args.triplet_weight * tri

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            n = y.size(0)
            tot["loss"] += loss.item() * n
            tot["arc"] += arc.item() * n
            tot["tri"] += tri.detach().item() * n
            tot["hit"] += (logits.argmax(1) == y).sum().item()
            tot["n"] += n

        n = max(1.0, tot["n"])
        m = validate(model, args.root, device, args.val_pairs, args.seed, args.workers)
        row = {
            "epoch": epoch,
            "loss": tot["loss"] / n,
            "arc_loss": tot["arc"] / n,
            "triplet_loss": tot["tri"] / n,
            "train_acc": tot["hit"] / n,
            "val_auc": m["roc_auc"],
            "val_eer": m["eer"],
            "val_best_acc": m["best_accuracy"],
            "val_threshold": m["best_accuracy_threshold"],
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"[{epoch:3d}/{args.epochs}] loss={row['loss']:.4f} "
              f"(arc={row['arc_loss']:.4f} tri={row['triplet_loss']:.4f}) "
              f"acc={row['train_acc']:.3f} | val AUC={row['val_auc']:.4f} "
              f"EER={row['val_eer']:.4f} | {row['seconds']:.0f}s")

        if m["roc_auc"] > best:
            best = m["roc_auc"]
            torch.save({"model": model.state_dict(), "dim": args.dim, "size": args.size,
                        "mask_p": args.mask_p,
                        "epoch": epoch,
                        "val_metrics": m, "backbone": "resnet50",
                        "train_identities": len(ds.classes)}, out / "best_model.pth")
            print(f"        saved (val AUC {best:.4f})")

    (out / "training_history.json").write_text(json.dumps({
        "backbone": "resnet50",
        "dim": args.dim,
        "size": args.size,
        "mask_p": args.mask_p,
        "train_identities": len(ds.classes),
        "train_images": len(ds),
        "history": history,
        "best_val_auc": best,
    }, indent=2))
    print(f"done, best val AUC = {best:.4f}")


if __name__ == "__main__":
    main()
