"""Audit the identity labels themselves.

Two failure modes matter for a face dataset:

  * one label, two people   - a name shared by different individuals, which turns
    some "genuine" pairs into impostor pairs
  * two labels, one person  - the same individual filed under two names, which
    turns some "impostor" pairs into genuine pairs

Both are detectable from the embeddings without any extra annotation: the first
shows up as a folder whose images split into separated clusters, the second as
two different folders whose templates sit unusually close together.

    python identity_audit.py --split test
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset_preparation import load_split
from model import embed, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--split-threshold", type=float, default=0.15,
                    help="mean intra-identity similarity below this is flagged")
    ap.add_argument("--merge-threshold", type=float, default=0.55,
                    help="cross-identity template similarity above this is flagged")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    names = {}
    man = Path(args.root) / "identity_manifest.json"
    if man.exists():
        names = {k: v["original_identity"].replace("_", " ")
                 for k, v in json.loads(man.read_text()).items()}

    people = load_split(args.root, args.split)
    folders = sorted(people)
    paths, owner = [], []
    for f in folders:
        paths += people[f]
        owner += [f] * len(people[f])

    print(f"embedding {len(paths)} images from {len(folders)} identities ...")
    z = embed(model, paths, device, 64)
    owner = np.array(owner)

    # ---- exact duplicate images ------------------------------------------- #
    digest = {}
    dupes = []
    for p in paths:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in digest:
            dupes.append((digest[h], p))
        else:
            digest[h] = p

    # ---- one label, two people -------------------------------------------- #
    coherence = []
    for f in folders:
        idx = np.where(owner == f)[0]
        if len(idx) < 2:
            continue
        s = z[idx] @ z[idx].T
        iu = np.triu_indices(len(idx), 1)
        vals = s[iu]
        coherence.append({
            "folder": f, "name": names.get(f, f), "images": int(len(idx)),
            "mean_intra": float(vals.mean()), "min_intra": float(vals.min()),
        })
    coherence.sort(key=lambda r: r["mean_intra"])
    suspect_split = [r for r in coherence if r["mean_intra"] < args.split_threshold]

    # ---- two labels, one person ------------------------------------------- #
    T = np.zeros((len(folders), z.shape[1]), np.float32)
    row = {f: i for i, f in enumerate(folders)}
    for v, o in zip(z, owner):
        T[row[o]] += v
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    cross = T @ T.T
    np.fill_diagonal(cross, -1)
    iu = np.triu_indices(len(folders), 1)
    pair_scores = cross[iu]
    top = np.argsort(-pair_scores)[: args.top]
    closest = [{
        "a": folders[iu[0][t]], "b": folders[iu[1][t]],
        "name_a": names.get(folders[iu[0][t]], folders[iu[0][t]]),
        "name_b": names.get(folders[iu[1][t]], folders[iu[1][t]]),
        "similarity": float(pair_scores[t]),
    } for t in top]
    suspect_merge = [c for c in closest if c["similarity"] >= args.merge_threshold]

    report = {
        "split": args.split,
        "identities": len(folders),
        "images": len(paths),
        "exact_duplicate_images": len(dupes),
        "split_threshold": args.split_threshold,
        "merge_threshold": args.merge_threshold,
        "suspect_one_label_two_people": suspect_split,
        "suspect_two_labels_one_person": suspect_merge,
        "lowest_coherence": coherence[: args.top],
        "closest_identity_pairs": closest,
    }
    (res / "identity_audit.json").write_text(json.dumps(report, indent=2))

    L = []
    P = L.append
    P("IDENTITY LABEL AUDIT")
    P(f"split: {args.split}   identities: {len(folders)}   images: {len(paths)}")
    P("")
    P("Two things can be wrong with identity labels, and both are detectable from")
    P("the embeddings alone:")
    P("")
    P("  A. ONE LABEL, TWO PEOPLE   - a folder holding more than one individual.")
    P("     Detected as low mean similarity between images inside that folder.")
    P("  B. TWO LABELS, ONE PERSON  - the same individual under two names.")
    P("     Detected as two folder templates sitting unusually close together.")
    P("")
    P("=" * 78)
    P("HOW THE PIPELINE HANDLES THEM")
    P("=" * 78)
    P("  Pair generation trusts the folder: same folder -> genuine, different")
    P("  folders -> impostor. So:")
    P("")
    P("    case A turns some GENUINE pairs into impostor pairs  -> scores too low")
    P("    case B turns some IMPOSTOR pairs into genuine pairs  -> scores too high")
    P("             for those pairs, i.e. they look like false accepts")
    P("")
    P("  Both push the measured numbers in the PESSIMISTIC direction: A depresses")
    P("  the genuine distribution, B inflates the apparent impostor tail. Neither")
    P("  can flatter the model - a label error cannot invent a correct match.")
    P("")
    P("  Exact duplicate images are checked separately and rejected outright:")
    P(f"    exact duplicate images found in this split: {len(dupes)}")
    P("    (check_leakage.py runs the same test across the whole dataset)")
    P("")
    P("=" * 78)
    P("A. LOWEST INTRA-IDENTITY COHERENCE  (candidates for one label, two people)")
    P("=" * 78)
    P(f"  {'mean':>7} {'min':>8}  {'imgs':>5}  identity")
    for r in coherence[: args.top]:
        flag = "  <- FLAGGED" if r["mean_intra"] < args.split_threshold else ""
        P(f"  {r['mean_intra']:7.4f} {r['min_intra']:8.4f}  {r['images']:>5}  {r['name']}{flag}")
    P("")
    P(f"  flagged below mean {args.split_threshold}: {len(suspect_split)}")
    P("  Low coherence usually means pose, age or lighting spread rather than a")
    P("  genuine label error - inspect before concluding anything.")
    P("")
    P("=" * 78)
    P("B. CLOSEST DISTINCT IDENTITIES  (candidates for two labels, one person)")
    P("=" * 78)
    P(f"  {'sim':>7}  identity A  vs  identity B")
    for c in closest:
        flag = "  <- FLAGGED" if c["similarity"] >= args.merge_threshold else ""
        P(f"  {c['similarity']:7.4f}  {c['name_a']}  vs  {c['name_b']}{flag}")
    P("")
    P(f"  flagged above {args.merge_threshold}: {len(suspect_merge)}")
    P("  High cross-identity similarity is usually a look-alike, and on LFW often")
    P("  a relative - the dataset contains several members of the same family.")
    P("  A true duplicate identity would show similarity close to the genuine mean.")

    txt = res / "identity_audit.txt"
    txt.write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n".join(L[-0:]) if False else "")
    print(f"exact duplicate images   : {len(dupes)}")
    print(f"flagged one-label-two-people : {len(suspect_split)}")
    print(f"flagged two-labels-one-person: {len(suspect_merge)}")
    print(f"\nwrote {txt} and identity_audit.json")


if __name__ == "__main__":
    main()
