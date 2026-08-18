"""Export the 512-D embeddings for a split.

Writes an .npz (embeddings + image paths + identity labels) plus an optional
plain-text dump, so the embeddings can be reused without re-running the model.

    python export_embeddings.py --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset_preparation import load_split
from model import embed, load_model


def export_mlfw(out, text, cache=Path("results/_mlfw_emb.npz")):
    """Re-label the MLFW vectors that dataset_roc.py scored.

    These are the exact embeddings the reported MLFW ROC was computed from, so
    the file matches the published numbers rather than a fresh re-run of them.
    Identity comes off the filename: Name_0001_0000.jpg -> Name.
    """
    from dataset_roc import mlfw_identity

    if not cache.exists():
        raise SystemExit(f"{cache} not found - run dataset_roc.py first")

    d = np.load(cache, allow_pickle=True)
    z = d["z"].astype(np.float32)
    img = np.array([Path(p).name for p in d["paths"]])
    ident = np.array([mlfw_identity(n) for n in img])
    order = np.lexsort((img, ident))
    z, img, ident = z[order], img[order], ident[order]

    npz = out / "embeddings_mlfw.npz"
    np.savez_compressed(npz, embeddings=z, image=img, identity=ident)
    print(f"[mlfw] {len(img)} images / {len(set(ident))} identities")
    print(f"  -> {npz}  {z.shape}  ({npz.stat().st_size/1e6:.2f} MB)")

    if text:
        txt = out / "embeddings_mlfw.txt"
        with txt.open("w", encoding="utf-8") as f:
            f.write(f"# {len(img)} L2-normalised {z.shape[1]}-D embeddings, MLFW (masked)\n")
            f.write("# identities absent from the training split only; origin/ images\n")
            f.write("# passed through the same YuNet + landmark alignment as LFW\n")
            f.write(f"# image<TAB>identity<TAB>v1 v2 ... v{z.shape[1]} (space separated)\n")
            for i, idn, v in zip(img, ident, z):
                f.write(f"{i}\t{idn}\t" + " ".join(f"{x:.6f}" for x in v) + "\n")
        print(f"  -> {txt}  ({txt.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--out-dir", default="results/embeddings")
    ap.add_argument("--text", action="store_true", help="also write a plain-text dump")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mlfw", action="store_true",
                    help="export the MLFW embeddings behind dataset_roc.py instead")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.mlfw:
        export_mlfw(out, args.text)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)

    names = {}
    man = Path(args.root) / "identity_manifest.json"
    if man.exists():
        names = {k: v["original_identity"] for k, v in json.loads(man.read_text()).items()}

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        people = load_split(args.root, split)
        paths, folders = [], []
        for folder, imgs in people.items():
            paths += imgs
            folders += [folder] * len(imgs)

        print(f"[{split}] embedding {len(paths)} images from {len(people)} identities ...")
        z = embed(model, paths, device, args.batch_size, args.workers)

        rel = [p.relative_to(args.root).as_posix() for p in paths]
        ident = [names.get(f, f) for f in folders]
        npz = out / f"embeddings_{split}.npz"
        np.savez_compressed(npz, embeddings=z.astype(np.float32),
                            image=np.array(rel), folder=np.array(folders),
                            identity=np.array(ident))
        print(f"  -> {npz}  {z.shape}  ({npz.stat().st_size/1e6:.2f} MB)")

        if args.text:
            txt = out / f"embeddings_{split}.txt"
            with txt.open("w", encoding="utf-8") as f:
                f.write(f"# {len(paths)} L2-normalised {z.shape[1]}-D embeddings, {split} split\n")
                f.write("# image<TAB>folder<TAB>identity<TAB>v1 v2 ... "
                        f"v{z.shape[1]} (space separated)\n")
                for r, fo, idn, v in zip(rel, folders, ident, z):
                    f.write(f"{r}\t{fo}\t{idn}\t" + " ".join(f"{x:.6f}" for x in v) + "\n")
            print(f"  -> {txt}  ({txt.stat().st_size/1e6:.1f} MB)")

    print("\nload with:  d = np.load('embeddings_test.npz'); d['embeddings'], d['identity']")
    print("embeddings are unit-norm, so cosine similarity is just a dot product.")


if __name__ == "__main__":
    main()
