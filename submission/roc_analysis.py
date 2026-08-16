"""ROC / AUC / EER analysis and threshold selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve


def eer_point(fpr, tpr, thr):
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def tar_at_far(fpr, tpr, thr, target):
    ok = np.where(fpr <= target)[0]
    if ok.size == 0:
        return 0.0, float(thr[0])
    i = int(ok[-1])
    return float(tpr[i]), float(thr[i])


def best_threshold(scores, labels):
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())

    tp = np.concatenate([[n_pos], n_pos - np.cumsum(y)])
    fp = np.concatenate([[n_neg], n_neg - np.cumsum(1 - y)])
    acc = (tp + (n_neg - fp)) / len(y)

    i = int(np.argmax(acc))
    if i == 0:
        t = float(s[0] - 1e-6)
    elif i >= len(s):
        t = float(s[-1] + 1e-6)
    else:
        t = float((s[i - 1] + s[i]) / 2)
    return t, float(acc[i])


def metrics(scores, labels, threshold=None):
    fpr, tpr, thr = roc_curve(labels, scores)
    eer, eer_thr = eer_point(fpr, tpr, thr)
    best_thr, best_acc = best_threshold(scores, labels)

    t = best_thr if threshold is None else float(threshold)
    pred = (scores >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)

    tar1, thr1 = tar_at_far(fpr, tpr, thr, 0.01)
    tar01, thr01 = tar_at_far(fpr, tpr, thr, 0.001)
    gen, imp = scores[labels == 1], scores[labels == 0]

    return {
        "roc_auc": float(auc(fpr, tpr)),
        "eer": eer,
        "eer_threshold": eer_thr,
        "best_accuracy_threshold": best_thr,
        "best_accuracy": best_acc,
        "chosen_threshold": t,
        "accuracy_at_chosen_threshold": float((pred == labels).mean()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tar_at_far_1pct": tar1,
        "threshold_at_far_1pct": thr1,
        "tar_at_far_0.1pct": tar01,
        "threshold_at_far_0.1pct": thr01,
        "false_accepts": int(((pred == 1) & (labels == 0)).sum()),
        "false_rejects": int(((pred == 0) & (labels == 1)).sum()),
        "num_genuine": int(len(gen)),
        "num_impostor": int(len(imp)),
        "genuine_mean": float(gen.mean()),
        "genuine_std": float(gen.std()),
        "impostor_mean": float(imp.mean()),
        "impostor_std": float(imp.std()),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def read_scores(path):
    df = pd.read_csv(path)
    return df["cosine_similarity"].to_numpy(), df["label"].to_numpy().astype(int)


def plot_roc(m, path):
    fpr = np.array(m["roc_curve"]["fpr"])
    tpr = np.array(m["roc_curve"]["tpr"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    ax[0].plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC (AUC = {m['roc_auc']:.4f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
    ax[0].plot(m["eer"], 1 - m["eer"], "o", color="#d62728", ms=8, label=f"EER = {m['eer']:.4f}")
    ax[0].set_xlabel("False Positive Rate (FAR)")
    ax[0].set_ylabel("True Positive Rate (TAR)")
    ax[0].set_title("ROC curve - face verification")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)

    keep = fpr > 0
    ax[1].semilogx(fpr[keep], tpr[keep], lw=2, color="#1f77b4")
    for far, key in ((0.01, "tar_at_far_1pct"), (0.001, "tar_at_far_0.1pct")):
        ax[1].axvline(far, ls="--", color="grey", lw=1)
        ax[1].plot(far, m[key], "o", ms=7, label=f"TAR@FAR={far:g} = {m[key]:.4f}")
    ax[1].set_xlabel("False Positive Rate (log)")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("Low-FAR operating region")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(scores, labels, t, path):
    gen, imp = scores[labels == 1], scores[labels == 0]
    plt.figure(figsize=(9, 5.5))
    bins = np.linspace(min(scores.min(), -0.2), 1.0, 80)
    plt.hist(imp, bins=bins, alpha=0.65, color="#d62728", density=True,
             label=f"Impostor (n={len(imp)}, mean={imp.mean():.3f})")
    plt.hist(gen, bins=bins, alpha=0.65, color="#2ca02c", density=True,
             label=f"Genuine (n={len(gen)}, mean={gen.mean():.3f})")
    plt.axvline(t, color="black", ls="--", lw=2, label=f"Threshold = {t:.4f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Genuine vs impostor similarity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_sweep(scores, labels, t, path):
    grid = np.linspace(scores.min(), scores.max(), 400)
    gen, imp = scores[labels == 1], scores[labels == 0]
    far = [(imp >= x).mean() for x in grid]
    frr = [(gen < x).mean() for x in grid]
    acc = [((scores >= x) == labels).mean() for x in grid]

    plt.figure(figsize=(9, 5.5))
    plt.plot(grid, far, color="#d62728", lw=2, label="FAR - false accepts")
    plt.plot(grid, frr, color="#1f77b4", lw=2, label="FRR - false rejects")
    plt.plot(grid, acc, color="#2ca02c", lw=2, label="Accuracy")
    plt.axvline(t, color="black", ls="--", lw=2, label=f"Chosen threshold = {t:.4f}")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Rate")
    plt.title("Threshold sweep")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/pair_scores.csv")
    ap.add_argument("--val-scores", default="results/pair_scores_val.csv")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    scores, labels = read_scores(args.scores)

    # threshold comes from validation so the test numbers stay honest
    val_path = Path(args.val_scores) if args.val_scores else None
    if val_path and val_path.exists():
        vm = metrics(*read_scores(val_path))
        vm.pop("roc_curve")
        t = vm["best_accuracy_threshold"]
        source = f"validation split ({val_path.name})"
    else:
        vm, t = None, None
        source = "test split (no validation scores given)"

    m = metrics(scores, labels, threshold=t)
    t = m["chosen_threshold"]

    plot_roc(m, res / "roc_curve.png")
    plot_distributions(scores, labels, t, res / "similarity_distribution.png")
    plot_sweep(scores, labels, t, res / "threshold_analysis.png")

    curve = m.pop("roc_curve")
    step = max(1, len(curve["fpr"]) // 500)
    report = {
        "verification_test": m,
        "threshold": {
            "value": t,
            "selected_on": source,
            "rule": "cosine_similarity >= threshold -> MATCH else NON-MATCH",
        },
        "roc_curve_points": {"fpr": curve["fpr"][::step], "tpr": curve["tpr"][::step]},
    }
    if vm:
        report["verification_validation"] = vm
    gp = res / "gallery_probe_metrics.json"
    if gp.exists():
        report["identification"] = json.loads(gp.read_text())

    (res / "evaluation_results.json").write_text(json.dumps(report, indent=2))

    print("\n--- verification (test) ---")
    print(f"  ROC-AUC          : {m['roc_auc']:.4f}")
    print(f"  EER              : {m['eer']:.4f} (thr {m['eer_threshold']:.4f})")
    print(f"  TAR @ FAR 1%     : {m['tar_at_far_1pct']:.4f}")
    print(f"  TAR @ FAR 0.1%   : {m['tar_at_far_0.1pct']:.4f}")
    print(f"  threshold        : {t:.4f} ({source})")
    print(f"  accuracy         : {m['accuracy_at_chosen_threshold']:.4f}")
    print(f"  P / R / F1       : {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}")
    print(f"  false accepts    : {m['false_accepts']} / {m['num_impostor']}")
    print(f"  false rejects    : {m['false_rejects']} / {m['num_genuine']}")
    if "identification" in report:
        i = report["identification"]
        print("\n--- identification ---")
        print(f"  Rank-1           : {i['rank1_accuracy']:.4f}")
        print(f"  Rank-5           : {i['rank5_accuracy']:.4f}")


if __name__ == "__main__":
    main()
