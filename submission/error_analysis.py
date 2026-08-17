"""List the pairs the system gets wrong at the operating threshold.

Writes the false accepts (impostor pairs scored at or above the threshold) and
the false rejects (genuine pairs scored below it), worst first, with the real
identity behind each `person_XXX` folder so the failures can be inspected.

    python error_analysis.py
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_names(root):
    path = Path(root) / "identity_manifest.json"
    if not path.exists():
        return {}
    man = json.loads(path.read_text())
    return {k: v["original_identity"].replace("_", " ") for k, v in man.items()}


def montage(rows, root, path, title, n=8):
    rows = rows[:n]
    if not rows:
        return
    fig, axes = plt.subplots(2, len(rows), figsize=(1.7 * len(rows), 4.2))
    axes = axes.reshape(2, -1)
    for col, r in enumerate(rows):
        for row, key in enumerate(("image_a", "image_b")):
            img = cv2.imread(str(Path(root) / r[key]))
            axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[row, col].axis("off")
        axes[0, col].set_title(f"{r['name_a']}\nvs {r['name_b']}\n{r['cosine_similarity']:.3f}",
                               fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--scores", default="results/pair_scores.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    res = Path(args.results_dir)
    ev = json.loads((res / "evaluation_results.json").read_text())
    thr = args.threshold if args.threshold is not None else ev["threshold"]["value"]

    names = load_names(args.root)
    df = pd.read_csv(args.scores)
    df["name_a"] = df["image_a"].map(lambda p: names.get(Path(p).parent.name, Path(p).parent.name))
    df["name_b"] = df["image_b"].map(lambda p: names.get(Path(p).parent.name, Path(p).parent.name))
    df["margin"] = (df["cosine_similarity"] - thr).round(4)

    fa = df[(df.label == 0) & (df.cosine_similarity >= thr)].sort_values(
        "cosine_similarity", ascending=False)
    fr = df[(df.label == 1) & (df.cosine_similarity < thr)].sort_values("cosine_similarity")

    cols = ["image_a", "image_b", "name_a", "name_b", "cosine_similarity", "margin"]
    fa[cols].to_csv(res / "false_accepts.csv", index=False)
    fr[cols].to_csv(res / "false_rejects.csv", index=False)

    n_imp = int((df.label == 0).sum())
    n_gen = int((df.label == 1).sum())
    summary = {
        "threshold": thr,
        "impostor_pairs": n_imp,
        "false_accepts": len(fa),
        "false_accept_rate": len(fa) / n_imp,
        "genuine_pairs": n_gen,
        "false_rejects": len(fr),
        "false_reject_rate": len(fr) / n_gen,
        "worst_false_accept": float(fa.cosine_similarity.max()) if len(fa) else None,
        "worst_false_reject": float(fr.cosine_similarity.min()) if len(fr) else None,
    }
    (res / "error_summary.json").write_text(json.dumps(summary, indent=2))

    montage(fa.to_dict("records"), args.root, res / "false_accepts_examples.png",
            f"Worst false accepts — different people scored as a match (threshold {thr:.4f})")
    montage(fr.to_dict("records"), args.root, res / "false_rejects_examples.png",
            f"Worst false rejects — same person scored as a non-match (threshold {thr:.4f})")

    print(f"threshold {thr:.4f}")
    print(f"  false accepts {len(fa):4d} / {n_imp} impostor pairs = {100*len(fa)/n_imp:.2f}% FAR")
    print(f"  false rejects {len(fr):4d} / {n_gen} genuine  pairs = {100*len(fr)/n_gen:.2f}% FRR")
    print("\nworst false accepts (different people, highest similarity):")
    for r in fa.head(5).to_dict("records"):
        print(f"  {r['cosine_similarity']:.4f}  {r['name_a']} vs {r['name_b']}")
    print("\nworst false rejects (same person, lowest similarity):")
    for r in fr.head(5).to_dict("records"):
        print(f"  {r['cosine_similarity']:.4f}  {r['name_a']}")
    print(f"\nwrote false_accepts.csv, false_rejects.csv, error_summary.json and montages -> {res}")


if __name__ == "__main__":
    main()
