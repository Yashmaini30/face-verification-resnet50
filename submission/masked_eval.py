"""Clean vs masked face verification.

MLFW measures how much a mask costs a recogniser. Its identities come from
CALFW, which draws from LFW - and 89% of LFW's multi-image identities are in this
project's training split, so a raw MLFW ROC would mostly measure memorisation.

Two modes:

  --synthetic   (default) draw a surgical mask over the lower face of this
                project's own held-out test identities, using the 5 landmarks
                already detected during preprocessing. No download, no leakage,
                because the model never saw these people.

  --mlfw DIR    score a real MLFW-style directory instead. Identities that also
                appear in the training split are excluded unless --no-filter is
                passed, and the exclusion is reported.

    python masked_eval.py --synthetic
    python masked_eval.py --mlfw ../data_raw/MLFW
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset_preparation import Cropper, load_split
from generate_pairs import make_pairs
from model import eval_tf, load_model
from roc_analysis import metrics

MASK_COLOURS = [(232, 232, 235), (176, 205, 214), (120, 145, 160), (245, 245, 248)]


def draw_mask(face, pts, colour):
    """Cover the lower face with a surgical-mask polygon built from the landmarks.

    pts are the 5 aligned landmarks in the cropped frame: right eye, left eye,
    nose, right mouth corner, left mouth corner.
    """
    h, w = face.shape[:2]
    (rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy) = pts
    eye_y = (rey + ley) / 2
    span = abs(lex - rex)
    top = ny - 0.10 * span                    # just under the nose bridge
    chin = min(h - 1, (rmy + lmy) / 2 + 1.05 * span)
    half = 0.92 * span

    cx = (rmx + lmx) / 2
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

    out = face.copy()
    cv2.fillPoly(out, [poly], colour, cv2.LINE_AA)
    # pleats and edge, so it reads as fabric rather than a flat blob
    for f in (0.35, 0.55, 0.75):
        y = int(top + f * (chin - top))
        cv2.line(out, (int(cx - half * 0.95), y), (int(cx + half * 0.95), y),
                 tuple(int(c * 0.92) for c in colour), 1, cv2.LINE_AA)
    cv2.polylines(out, [poly], True, tuple(int(c * 0.75) for c in colour), 2, cv2.LINE_AA)
    # straps towards the ears
    cv2.line(out, (int(cx - half), int(top + 0.15 * span)), (0, int(eye_y + 0.15 * span)),
             tuple(int(c * 0.8) for c in colour), 2, cv2.LINE_AA)
    cv2.line(out, (int(cx + half), int(top + 0.15 * span)), (w - 1, int(eye_y + 0.15 * span)),
             tuple(int(c * 0.8) for c in colour), 2, cv2.LINE_AA)
    return out


@torch.no_grad()
def embed_images(model, images, device, tf, batch=64):
    out = []
    for i in range(0, len(images), batch):
        xs = torch.stack([tf(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                          for im in images[i:i + batch]]).to(device)
        z = model(xs, normalize=False) + model(torch.flip(xs, [3]), normalize=False)
        out.append(F.normalize(z).cpu().numpy())
    return np.concatenate(out)


def score(model, images, index, pairs, device, tf):
    z = embed_images(model, images, device, tf)
    a = np.array([index[str(p)] for p, _, _ in pairs])
    b = np.array([index[str(p)] for _, p, _ in pairs])
    y = np.array([t for _, _, t in pairs])
    return np.sum(z[a] * z[b], axis=1), y


def mlfw_identity(filename):
    """Aaron_Eckhart_0001_0000.jpg -> Aaron_Eckhart"""
    return re.sub(r"_\d{4}_\d{4}\.jpg$", "", filename)


def run_mlfw(args, model, tf, device, res):
    """Score the real MLFW release, both on its official protocol and leakage-free.

    MLFW derives from CALFW, which draws from LFW, so a large share of its
    identities are people this model was trained on. The official 6,000-pair
    protocol is reported for comparability, and a second run over identities the
    model has never seen is reported as the number that actually generalises.
    """
    root = Path(args.mlfw)
    img_dir = root / "aligned" if (root / "aligned").is_dir() else root
    pair_file = root / "pairs.txt"

    man = json.loads((Path(args.root) / "identity_manifest.json").read_text())
    sp = json.loads((Path(args.root) / "splits.json").read_text())
    trained = {man[f]["original_identity"] for f in sp["train"]}

    rows = [l.split("\t") for l in pair_file.read_text().strip().split("\n") if l.strip()]
    official = [(img_dir / a, img_dir / b, int(y)) for a, b, y in rows]
    unseen = [(a, b, y) for a, b, y in official
              if not ({mlfw_identity(a.name), mlfw_identity(b.name)} & trained)]

    ids = {mlfw_identity(a.name) for a, _, _ in official} | \
          {mlfw_identity(b.name) for _, b, _ in official}
    overlap = len(ids & trained)
    print(f"MLFW: {len(official):,} official pairs, {len(ids):,} identities")
    print(f"  {overlap:,} identities ({100*overlap/len(ids):.1f}%) are in this model's TRAIN split")
    print(f"  leakage-free pairs: {len(unseen):,}")

    # a balanced leakage-free set, built from identities the model never saw
    by_id = {}
    for p in sorted(img_dir.glob("*.jpg")):
        who = mlfw_identity(p.name)
        if who not in trained:
            by_id.setdefault(who, []).append(p)
    by_id = {k: v for k, v in by_id.items() if len(v) >= 2}
    balanced, bstats = make_pairs(by_id, args.n_pairs // 2, args.n_pairs // 2, args.seed)
    print(f"  balanced leakage-free set: {bstats['generated_positive']} genuine + "
          f"{bstats['generated_negative']} impostor from {len(by_id):,} unseen identities")

    out = {"source": f"MLFW ({img_dir.name} images)",
           "official_pairs": len(official), "identities": len(ids),
           "identities_in_train": overlap,
           "train_overlap_fraction": overlap / len(ids),
           "leakage_free_official_pairs": len(unseen)}
    curves = []

    for name, subset in [("official (leaky)", official),
                         ("leakage-free subset", unseen),
                         ("leakage-free balanced", balanced)]:
        if len(subset) < 50:
            continue
        paths = sorted({str(p) for a, b, _ in subset for p in (a, b)})
        index = {p: i for i, p in enumerate(paths)}
        images = [cv2.imread(p) for p in paths]
        s, y = score(model, images, index, subset, device, tf)
        m = metrics(s, y)
        curves.append((name, m))
        out[name.replace(" ", "_")] = {
            "pairs": len(subset), "genuine": int(y.sum()), "impostor": int((y == 0).sum()),
            **{k: m[k] for k in ("roc_auc", "eer", "best_accuracy",
                                 "tar_at_far_1pct", "tar_at_far_0.1pct")}}
        print(f"  {name:24s} AUC {m['roc_auc']:.4f}  EER {m['eer']*100:5.2f}%  "
              f"acc {m['best_accuracy']*100:5.2f}%  ({len(subset):,} pairs)")

    plt.figure(figsize=(7.5, 6))
    for name, m in curves:
        fpr = np.array(m["roc_curve"]["fpr"]); tpr = np.array(m["roc_curve"]["tpr"])
        plt.plot(fpr, tpr, lw=2,
                 label=f"{name}  AUC {m['roc_auc']:.4f}  EER {m['eer']*100:.2f}%")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TAR)")
    plt.title("ROC on MLFW (masked faces)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(res / "mlfw_roc.png", dpi=150)
    plt.close()

    (res / "mlfw_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote mlfw_roc.png and mlfw_metrics.json -> {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--mlfw", default=None, help="directory of a real MLFW-style dataset")
    ap.add_argument("--no-filter", action="store_true",
                    help="do NOT exclude MLFW identities seen in training")
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    tf = eval_tf(model.input_size)
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    if args.mlfw:
        return run_mlfw(args, model, tf, device, res)

    if True:
        source = f"synthetic masks on the {args.split} split"
        people = load_split(args.root, args.split)
        lm = json.loads((Path(args.root) / "landmarks.json").read_text())
        cropper = Cropper(size=model.input_size)
        rng = np.random.default_rng(args.seed)

        clean, masked = {}, {}
        for folder, imgs in people.items():
            for p in imgs:
                img = cv2.imread(str(p))
                clean[str(p)] = img
                pts = cropper.landmarks(img)
                if pts is None:
                    masked[str(p)] = img          # nothing to mask; keep it comparable
                else:
                    masked[str(p)] = draw_mask(img, pts,
                                               MASK_COLOURS[rng.integers(len(MASK_COLOURS))])
        print(f"masked {len(masked)} images across {len(people)} identities")

    pairs, stats = make_pairs(people, args.n_pairs // 2, args.n_pairs // 2, args.seed)
    print(f"{stats['generated_positive']} genuine + {stats['generated_negative']} impostor pairs")

    keys = sorted(clean)
    index = {k: i for i, k in enumerate(keys)}
    s_clean, y = score(model, [clean[k] for k in keys], index, pairs, device, tf)
    m_clean = metrics(s_clean, y)

    out = {"source": source, "pairs": len(pairs),
           "clean": {k: m_clean[k] for k in ("roc_auc", "eer", "best_accuracy",
                                             "tar_at_far_1pct", "tar_at_far_0.1pct")}}
    curves = [("clean", m_clean)]

    if masked is not None:
        s_mask, _ = score(model, [masked[k] for k in keys], index, pairs, device, tf)
        m_mask = metrics(s_mask, y)
        curves.append(("masked", m_mask))
        out["masked"] = {k: m_mask[k] for k in ("roc_auc", "eer", "best_accuracy",
                                                "tar_at_far_1pct", "tar_at_far_0.1pct")}
        out["auc_drop"] = m_clean["roc_auc"] - m_mask["roc_auc"]
        out["eer_increase"] = m_mask["eer"] - m_clean["eer"]

        grid = []
        for k in keys[:6]:
            grid.append(np.hstack([clean[k], masked[k]]))
        cv2.imwrite(str(res / "masked_examples.png"), np.vstack(grid[:3]))

    plt.figure(figsize=(7.5, 6))
    for name, m in curves:
        fpr = np.array(m["roc_curve"]["fpr"]); tpr = np.array(m["roc_curve"]["tpr"])
        plt.plot(fpr, tpr, lw=2, label=f"{name}  AUC {m['roc_auc']:.4f}  EER {m['eer']*100:.2f}%")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TAR)")
    plt.title(f"ROC — {source}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(res / "masked_roc.png", dpi=150)
    plt.close()

    (res / "masked_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\n  clean : AUC {m_clean['roc_auc']:.4f}  EER {m_clean['eer']*100:.2f}%")
    if masked is not None:
        print(f"  masked: AUC {m_mask['roc_auc']:.4f}  EER {m_mask['eer']*100:.2f}%")
        print(f"  cost of the mask: -{out['auc_drop']:.4f} AUC, "
              f"+{out['eer_increase']*100:.2f} pp EER")
    print(f"\nwrote masked_roc.png, masked_metrics.json -> {res}")


if __name__ == "__main__":
    main()
