"""Score verification pairs by cosine similarity between embeddings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import embed, load_model
from roc_analysis import metrics


def read_pairs(path, root):
    a, b, y = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a.append(Path(root) / row["image_a"])
            b.append(Path(root) / row["image_b"])
            y.append(int(row["label"]))
    return a, b, np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pairs", default="results/pairs_test.csv")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--out", default="results/pair_scores.csv")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)

    a, b, y = read_pairs(args.pairs, args.root)
    print(f"{len(y)} pairs ({y.sum()} genuine / {(y == 0).sum()} impostor)")

    paths = sorted(set(a + b))
    idx = {p: i for i, p in enumerate(paths)}
    print(f"embedding {len(paths)} unique images on {device}")
    z = embed(model, paths, device, args.batch_size, args.workers)

    za = z[[idx[p] for p in a]]
    zb = z[[idx[p] for p in b]]
    scores = np.sum(za * zb, axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "label", "cosine_similarity"])
        for pa, pb, label, s in zip(a, b, y, scores):
            w.writerow([pa.relative_to(args.root).as_posix(),
                        pb.relative_to(args.root).as_posix(), int(label), f"{s:.6f}"])

    m = metrics(scores, y)
    m.pop("roc_curve")
    out.with_name("pair_score_summary.json").write_text(json.dumps(m, indent=2))

    print(f"genuine  {m['genuine_mean']:.4f} +/- {m['genuine_std']:.4f}")
    print(f"impostor {m['impostor_mean']:.4f} +/- {m['impostor_std']:.4f}")
    print(f"AUC={m['roc_auc']:.4f}  EER={m['eer']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
