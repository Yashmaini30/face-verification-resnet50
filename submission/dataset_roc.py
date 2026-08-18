"""ROC per dataset: LFW against MLFW, side by side.

This characterises the two DATASETS under one fixed model, rather than comparing
models. The same checkpoint, the same threshold and the same balanced pairing
protocol are applied to both, so the gap between the curves is the cost of the
mask and nothing else.

MLFW derives from CALFW, which draws from LFW, so a large share of its
identities are people this model trained on. Only identities absent from the
training split are used, and the exclusion is reported on the figure.

    python dataset_roc.py --mlfw ../data_raw/MLFW
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

from dataset_preparation import Cropper
from generate_pairs import make_pairs
from model import eval_tf, load_model
from roc_analysis import metrics


def all_pairs(z, owner, block=2048):
    """Every C(n,2) pair from an embedding matrix, blocked to bound memory.

    The sampled pair set cannot resolve a FAR below 1/n_impostor. Scoring every
    pair instead pushes that floor down by orders of magnitude, which is what the
    low-FAR panel needs to be smooth rather than a staircase of single pairs.
    """
    owner = np.asarray(owner)
    n = len(z)
    scores, labels = [], []
    for i in range(0, n, block):
        hi = min(i + block, n)
        sim = z[i:hi] @ z.T
        same = owner[i:hi, None] == owner[None, :]
        for r in range(hi - i):
            j0 = i + r + 1                      # upper triangle only
            if j0 >= n:
                continue
            scores.append(sim[r, j0:])
            labels.append(same[r, j0:])
    return np.concatenate(scores), np.concatenate(labels).astype(int)


def mlfw_identity(name):
    return re.sub(r"_\d{4}_\d{4}\.jpg$", "", name)


@torch.no_grad()
def embed_files(model, paths, device, tf, batch=64, cropper=None):
    """Embed images, optionally running them through our own detect+align first.

    MLFW ships 112x112 crops made with its own alignment. Feeding those to a model
    trained on 224x224 crops from our YuNet landmark pipeline compares two
    preprocessing chains, not two datasets. Passing a cropper here applies the
    identical pipeline to both sides.
    """
    out = []
    for i in range(0, len(paths), batch):
        ims = []
        for p in paths[i:i + batch]:
            im = cv2.imread(str(p))
            if cropper is not None:
                face = cropper.from_array(im, fallback=True)
                if face is not None:
                    im = face
            ims.append(im)
        xs = torch.stack([tf(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                          for im in ims]).to(device)
        z = model(xs, normalize=False) + model(torch.flip(xs, [3]), normalize=False)
        out.append(F.normalize(z).cpu().numpy())
    return np.concatenate(out)


def score_pairs(model, pairs, device, tf, cache=None, cropper=None):
    paths = sorted({str(p) for a, b, _ in pairs for p in (a, b)})
    if cache is not None and cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == paths:
            z, index = d["z"], {p: i for i, p in enumerate(paths)}
        else:
            z = None
    else:
        z = None
    if z is None:
        print(f"    embedding {len(paths):,} images ...")
        z = embed_files(model, [Path(p) for p in paths], device, tf, cropper=cropper)
        if cache is not None:
            np.savez_compressed(cache, z=z, paths=np.array(paths))
    index = {p: i for i, p in enumerate(paths)}
    a = np.array([index[str(p)] for p, _, _ in pairs])
    b = np.array([index[str(p)] for _, p, _ in pairs])
    y = np.array([t for _, _, t in pairs])
    return np.sum(z[a] * z[b], axis=1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--mlfw", default="../data_raw/MLFW")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--lfw-scores", default="results/pair_scores.csv")
    ap.add_argument("--n-pairs", type=int, default=10000)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    tf = eval_tf(model.input_size)
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    # ---- LFW: reuse the scores already computed for the test split --------- #
    import pandas as pd
    df = pd.read_csv(args.lfw_scores)
    lfw_s = df["cosine_similarity"].to_numpy()
    lfw_y = df["label"].to_numpy().astype(int)
    m_lfw = metrics(lfw_s, lfw_y)
    print(f"LFW  : {len(lfw_y):,} sampled pairs   AUC {m_lfw['roc_auc']:.4f}  "
          f"EER {m_lfw['eer']*100:.2f}%")

    lfw_emb = Path(args.results_dir) / "embeddings/embeddings_test.npz"
    m_lfw_all = None
    if lfw_emb.exists():
        d = np.load(lfw_emb, allow_pickle=False)
        s_all, y_all = all_pairs(d["embeddings"], d["folder"])
        m_lfw_all = metrics(s_all, y_all)
        print(f"LFW  : {len(y_all):,} exhaustive pairs "
              f"({int((y_all==0).sum()):,} impostor)   AUC {m_lfw_all['roc_auc']:.4f}")

    # ---- MLFW: balanced pairs over identities never seen in training ------- #
    man = json.loads((Path(args.root) / "identity_manifest.json").read_text())
    sp = json.loads((Path(args.root) / "splits.json").read_text())
    trained = {man[f]["original_identity"] for f in sp["train"]}

    root = Path(args.mlfw)
    use_origin = (root / "origin").is_dir()
    img_dir = root / "origin" if use_origin else (
        root / "aligned" if (root / "aligned").is_dir() else root)
    cropper = Cropper(size=model.input_size) if use_origin else None
    print(f"MLFW : using {img_dir.name}/ images"
          + ("  (our own YuNet + landmark alignment applied)" if use_origin else ""))
    by_id = {}
    for p in sorted(img_dir.glob("*.jpg")):
        who = mlfw_identity(p.name)
        if who not in trained:
            by_id.setdefault(who, []).append(p)
    by_id = {k: v for k, v in by_id.items() if len(v) >= 2}
    total_ids = len({mlfw_identity(p.name) for p in img_dir.glob("*.jpg")})
    print(f"MLFW : {len(by_id):,} usable identities never seen in training "
          f"(of {total_ids:,} total)")

    pairs, st = make_pairs(by_id, args.n_pairs // 2, args.n_pairs // 2, args.seed)
    mlfw_s, mlfw_y = score_pairs(model, pairs, device, tf, res / "_mlfw_emb.npz", cropper)
    m_mlfw = metrics(mlfw_s, mlfw_y)
    print(f"MLFW : {len(mlfw_y):,} sampled pairs   AUC {m_mlfw['roc_auc']:.4f}  "
          f"EER {m_mlfw['eer']*100:.2f}%")

    cache = res / "_mlfw_emb.npz"
    m_mlfw_all = None
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        owner = [mlfw_identity(Path(p).name) for p in d["paths"]]
        s_all, y_all = all_pairs(d["z"], owner)
        m_mlfw_all = metrics(s_all, y_all)
        print(f"MLFW : {len(y_all):,} exhaustive pairs "
              f"({int((y_all==0).sum()):,} impostor)   AUC {m_mlfw_all['roc_auc']:.4f}")

    # ---- side-by-side figure ---------------------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for name, m, colour in [("LFW (unmasked)", m_lfw, "#1f77b4"),
                            ("MLFW (masked)", m_mlfw, "#d62728")]:
        fpr = np.array(m["roc_curve"]["fpr"])
        tpr = np.array(m["roc_curve"]["tpr"])
        ax[0].plot(fpr, tpr, lw=2.2, color=colour,
                   label=f"{name}   AUC {m['roc_auc']:.4f}")
        ax[0].plot(m["eer"], 1 - m["eer"], "o", color=colour, ms=7)


    for name, m, colour in [("LFW (unmasked)", m_lfw_all or m_lfw, "#1f77b4"),
                            ("MLFW (masked)", m_mlfw_all or m_mlfw, "#d62728")]:
        fpr = np.array(m["roc_curve"]["fpr"])
        tpr = np.array(m["roc_curve"]["tpr"])
        keep = fpr > 0
        n_imp = m["num_impostor"]
        ax[1].semilogx(fpr[keep], tpr[keep], lw=2.2, color=colour,
                       label=f"{name}   {n_imp:,} impostor pairs")
        for far in (1e-4, 1e-3, 1e-2):
            ok = np.where(fpr <= far)[0]
            if ok.size:
                ax[1].plot(far, tpr[ok[-1]], "o", color=colour, ms=6)

    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax[0].set_xlabel("False Positive Rate (FAR)")
    ax[0].set_ylabel("True Positive Rate (TAR)")
    ax[0].set_title("ROC by dataset — same model, same protocol")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)

    for far in (1e-4, 1e-3, 1e-2):
        ax[1].axvline(far, ls="--", color="grey", lw=1)
    ax[1].set_xlim(left=5e-6)
    ax[1].set_xlabel("False Positive Rate (log scale)")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("Low-FAR region - every pair scored, not a 5k sample")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3, which="both")

    fig.suptitle(
        f"LFW vs MLFW — ResNet50 + ArcFace, identities unseen in training "
        f"({len(by_id):,} MLFW identities kept of {total_ids:,})", fontsize=11)
    fig.tight_layout()
    fig.savefig(res / "dataset_roc_lfw_vs_mlfw.png", dpi=150)
    plt.close(fig)

    keys = ("roc_auc", "eer", "best_accuracy", "tar_at_far_1pct", "tar_at_far_0.1pct")
    out = {
        "note": "same checkpoint and pairing protocol applied to both datasets; "
                "MLFW restricted to identities absent from the training split",
        "LFW": {"pairs": int(len(lfw_y)), **{k: m_lfw[k] for k in keys},
                "exhaustive": ({"pairs": m_lfw_all["num_genuine"] + m_lfw_all["num_impostor"],
                                "impostor_pairs": m_lfw_all["num_impostor"],
                                **{k: m_lfw_all[k] for k in keys}} if m_lfw_all else None)},
        "MLFW": {"pairs": int(len(mlfw_y)), "identities_used": len(by_id),
                 "identities_total": total_ids, "images": img_dir.name,
                 "preprocessing": "our YuNet + landmark alignment" if use_origin
                                  else "MLFW's own 112px alignment",
                 **{k: m_mlfw[k] for k in keys},
                 "exhaustive": ({"pairs": m_mlfw_all["num_genuine"] + m_mlfw_all["num_impostor"],
                                 "impostor_pairs": m_mlfw_all["num_impostor"],
                                 **{k: m_mlfw_all[k] for k in keys}} if m_mlfw_all else None)},
        "cost_of_mask": {
            "auc_drop": m_lfw["roc_auc"] - m_mlfw["roc_auc"],
            "eer_increase_pp": (m_mlfw["eer"] - m_lfw["eer"]) * 100,
            "tar_at_far_1pct_drop_pp": (m_lfw["tar_at_far_1pct"] - m_mlfw["tar_at_far_1pct"]) * 100,
        },
    }
    (res / "dataset_roc_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\ncost of the mask: -{out['cost_of_mask']['auc_drop']:.4f} AUC, "
          f"+{out['cost_of_mask']['eer_increase_pp']:.2f} pp EER, "
          f"-{out['cost_of_mask']['tar_at_far_1pct_drop_pp']:.2f} pp TAR@FAR=1%")
    print(f"wrote dataset_roc_lfw_vs_mlfw.png and dataset_roc_metrics.json -> {res}")


if __name__ == "__main__":
    main()
