"""Rebuild the demo's pre-enrolled gallery templates.

The demo ships 120 test identities already embedded, so it does not need the
dataset at runtime. Those templates and the checkpoint must come from the same
model - mixing them silently destroys identification, since two embedding
spaces have no shared geometry. Re-run this whenever the checkpoint changes.

    python demo/build_gallery.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))

from dataset_preparation import load_split          # noqa: E402
from model import embed, load_model                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "submission"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default=str(ROOT / "submission/checkpoints/best_model.pth"))
    ap.add_argument("--per-id", type=int, default=2, help="images enrolled per identity")
    ap.add_argument("--out", default=str(Path(__file__).parent / "gallery/gallery.npz"))
    args = ap.parse_args()

    out = Path(args.out)
    keep = None
    if out.exists():                    # preserve order so thumbs stay aligned
        old = np.load(out, allow_pickle=False)
        keep = (list(old["folders"]), list(old["labels"]))

    model = load_model(args.checkpoint, "cpu")
    people = load_split(args.root, args.split)
    folders = keep[0] if keep else sorted(people)
    labels = keep[1] if keep else folders

    templates = []
    for f in folders:
        z = embed(model, people[f][:args.per_id], "cpu", 32, 0)
        t = z.mean(0)
        templates.append(t / np.linalg.norm(t))
    T = np.stack(templates).astype(np.float32)

    np.savez_compressed(out, templates=T, folders=np.array(folders),
                        labels=np.array(labels))
    print(f"{len(folders)} identities x {args.per_id} images -> {out} "
          f"({out.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
