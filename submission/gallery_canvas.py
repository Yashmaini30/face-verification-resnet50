"""Render the gallery/probe search as a single image.

For each probe: the probe face on the left, then the top-k gallery matches with
their similarity, correct match outlined green and wrong ones red. Makes the
identification result inspectable without reading a CSV.

    python gallery_canvas.py --num-probes 8 --top-k 5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset_preparation import load_split
from model import embed, load_model

GREEN, RED, GREY, WHITE = (76, 175, 80), (211, 47, 47), (150, 150, 150), (255, 255, 255)


def label(canvas, text, x, y, scale=0.42, colour=(40, 40, 40), thick=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--gallery-per-id", type=int, default=1)
    ap.add_argument("--num-probes", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cell", type=int, default=104)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gallery_probe_canvas.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)

    names = {}
    man = Path(args.root) / "identity_manifest.json"
    if man.exists():
        names = {k: v["original_identity"].replace("_", " ")
                 for k, v in json.loads(man.read_text()).items()}

    people = load_split(args.root, args.split)
    g_paths, g_who, p_paths, p_who = [], [], [], []
    for person, imgs in people.items():
        if len(imgs) <= args.gallery_per_id:
            continue
        g_paths += imgs[:args.gallery_per_id]
        g_who += [person] * args.gallery_per_id
        p_paths += imgs[args.gallery_per_id:]
        p_who += [person] * (len(imgs) - args.gallery_per_id)

    ids = sorted(set(g_who))
    row_of = {p: i for i, p in enumerate(ids)}
    print(f"{len(ids)} identities enrolled, {len(p_paths)} probes")

    zg = embed(model, g_paths, device, 64)
    T = np.zeros((len(ids), zg.shape[1]), np.float32)
    for v, who in zip(zg, g_who):
        T[row_of[who]] += v
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    # a mix of hits and misses is more informative than a random sample
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(p_paths), size=min(args.num_probes * 4, len(p_paths)), replace=False)
    zp = embed(model, [p_paths[i] for i in pick], device, 64)
    sim = zp @ T.T
    order = np.argsort(-sim, axis=1)
    truth = np.array([row_of[p_who[i]] for i in pick])
    hit = order[:, 0] == truth

    chosen = list(np.where(hit)[0][: args.num_probes // 2]) + \
             list(np.where(~hit)[0][: args.num_probes - args.num_probes // 2])
    chosen = chosen[: args.num_probes]
    if not chosen:
        chosen = list(range(min(args.num_probes, len(pick))))

    c = args.cell
    pad, head, gap = 8, 34, 26
    W = pad + c + gap + (args.top_k * (c + pad)) + pad
    H = head + len(chosen) * (c + gap) + pad
    canvas = np.full((H, W, 3), 250, np.uint8)

    label(canvas, "PROBE", pad, 22, 0.5, (60, 60, 60), 1)
    label(canvas, f"TOP-{args.top_k} GALLERY MATCHES   (green = correct identity)",
          pad + c + gap, 22, 0.5, (60, 60, 60), 1)

    for r, k in enumerate(chosen):
        y = head + r * (c + gap)
        img = cv2.resize(cv2.imread(str(p_paths[pick[k]])), (c, c))
        canvas[y:y + c, pad:pad + c] = img
        cv2.rectangle(canvas, (pad, y), (pad + c, y + c), GREY, 1)
        label(canvas, names.get(p_who[pick[k]], p_who[pick[k]])[:18], pad, y + c + 14, 0.40)

        for j in range(args.top_k):
            gi = order[k, j]
            x = pad + c + gap + j * (c + pad)
            thumb = cv2.resize(cv2.imread(str(g_paths[g_who.index(ids[gi])])), (c, c))
            canvas[y:y + c, x:x + c] = thumb
            ok = gi == truth[k]
            cv2.rectangle(canvas, (x, y), (x + c, y + c), GREEN if ok else RED, 2 if ok else 1)
            label(canvas, f"{sim[k, gi]:.3f}", x, y + c + 14, 0.42,
                  (30, 110, 40) if ok else (120, 40, 40), 1)
            label(canvas, names.get(ids[gi], ids[gi])[:18], x, y + c + 24, 0.34)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}  ({canvas.shape[1]}x{canvas.shape[0]}), "
          f"{int(hit[chosen].sum())} of {len(chosen)} shown probes are rank-1 correct")


if __name__ == "__main__":
    main()
