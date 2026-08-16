"""Closed-set identification: enrol a gallery, match probes by cosine similarity."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset_preparation import load_split
from model import embed, load_model


def split_gallery(people, n_gallery):
    g_paths, g_who, p_paths, p_who = [], [], [], []
    for person, paths in people.items():
        if len(paths) <= n_gallery:
            continue
        g_paths += paths[:n_gallery]
        g_who += [person] * n_gallery
        p_paths += paths[n_gallery:]
        p_who += [person] * (len(paths) - n_gallery)
    return g_paths, g_who, p_paths, p_who


def identify(z_of, people, n_gallery, max_rank):
    g_paths, g_who, p_paths, p_who = split_gallery(people, n_gallery)
    names = sorted(set(g_who))
    row = {p: i for i, p in enumerate(names)}

    templates = np.zeros((len(names), len(next(iter(z_of.values())))), dtype=np.float32)
    for path, person in zip(g_paths, g_who):
        templates[row[person]] += z_of[path]
    templates /= np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12

    zp = np.stack([z_of[p] for p in p_paths])
    sim = zp @ templates.T
    order = np.argsort(-sim, axis=1)
    truth = np.array([row[p] for p in p_who])
    rank = (order == truth[:, None]).argmax(axis=1) + 1

    depth = min(max_rank, len(names))
    cmc = [float((rank <= r).mean()) for r in range(1, depth + 1)]
    return {
        "gallery_per_identity": n_gallery,
        "num_identities": len(names),
        "num_gallery_images": len(g_paths),
        "num_probe_images": len(p_paths),
        "rank1_accuracy": cmc[0],
        "rank5_accuracy": cmc[min(4, depth - 1)],
        "chance_rank1": 1.0 / len(names),
        "mean_rank_of_true_identity": float(rank.mean()),
        "cmc": cmc,
    }, (names, p_paths, p_who, sim, order, truth, rank)


def open_set(z_of, people, n_gallery, n_unknown, max_rank):
    """Open-set protocol: some identities are never enrolled.

    Closed-set Rank-1 assumes every probe belongs to someone in the gallery, so
    the system can always answer. A deployed system also has to say "I don't
    know". Here the last `n_unknown` identities are withheld from the gallery
    and all of their images become impostor probes, so accepting one is a false
    alarm rather than a ranking mistake.
    """
    names = sorted(people)
    unknown_names = names[-n_unknown:]
    known = {n: people[n] for n in names if n not in set(unknown_names)}

    g_paths, g_who, k_paths, k_who = split_gallery(known, n_gallery)
    enrolled = sorted(set(g_who))
    row = {p: i for i, p in enumerate(enrolled)}

    dim = len(next(iter(z_of.values())))
    templates = np.zeros((len(enrolled), dim), dtype=np.float32)
    for path, person in zip(g_paths, g_who):
        templates[row[person]] += z_of[path]
    templates /= np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12

    known_sim = np.stack([z_of[p] for p in k_paths]) @ templates.T
    known_top = known_sim.max(axis=1)
    known_hit = known_sim.argmax(axis=1) == np.array([row[p] for p in k_who])

    u_paths = [p for n in unknown_names for p in people[n]]
    unknown_top = (np.stack([z_of[p] for p in u_paths]) @ templates.T).max(axis=1)

    grid = np.unique(np.concatenate([known_top, unknown_top]))
    # DIR = probe is accepted AND ranked to the right person.
    # FPIR = an unknown probe is accepted at all.
    dir_at = np.array([float(((known_top >= t) & known_hit).mean()) for t in grid])
    fpir_at = np.array([float((unknown_top >= t).mean()) for t in grid])

    def dir_at_fpir(target):
        ok = np.where(fpir_at <= target)[0]
        if ok.size == 0:
            return 0.0, float(grid[-1])
        i = int(ok[0])  # fpir falls as the threshold rises
        return float(dir_at[i]), float(grid[i])

    dir_1, thr_1 = dir_at_fpir(0.01)
    dir_10, thr_10 = dir_at_fpir(0.10)
    return {
        "enrolled_identities": len(enrolled),
        "unknown_identities": len(unknown_names),
        "known_probes": len(k_paths),
        "unknown_probes": len(u_paths),
        "closed_set_rank1": float(known_hit.mean()),
        "dir_at_fpir_1pct": dir_1,
        "threshold_at_fpir_1pct": thr_1,
        "dir_at_fpir_10pct": dir_10,
        "threshold_at_fpir_10pct": thr_10,
        "known_top1_mean": float(known_top.mean()),
        "unknown_top1_mean": float(unknown_top.mean()),
        "curve": {"fpir": fpir_at.tolist()[::max(1, len(grid) // 300)],
                  "dir": dir_at.tolist()[::max(1, len(grid) // 300)]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--gallery-per-id", type=int, default=1)
    ap.add_argument("--sweep", type=int, default=0,
                    help="also evaluate gallery sizes 1..N and write a comparison")
    ap.add_argument("--open-set", type=int, default=0,
                    help="withhold N identities from the gallery and score them as unknowns")
    ap.add_argument("--max-rank", type=int, default=20)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    people = load_split(args.root, args.split)

    # embed every image in the split once and reuse it for each gallery size
    paths = sorted(p for v in people.values() for p in v)
    z = embed(model, paths, device, args.batch_size, args.workers)
    z_of = dict(zip(paths, z))

    out, detail = identify(z_of, people, args.gallery_per_id, args.max_rank)
    names, p_paths, p_who, sim, order, truth, rank = detail
    out["split"] = args.split
    print(f"{out['num_identities']} identities enrolled "
          f"({out['num_gallery_images']} images), {out['num_probe_images']} probes")

    n = np.arange(len(p_paths))
    pd.DataFrame({
        "probe_image": [p.relative_to(args.root).as_posix() for p in p_paths],
        "true_identity": p_who,
        "predicted_identity": [names[i] for i in order[:, 0]],
        "top1_similarity": sim[n, order[:, 0]],
        "true_identity_similarity": sim[n, truth],
        "rank_of_true_identity": rank,
        "correct": rank == 1,
    }).to_csv(res / "gallery_probe_results.csv", index=False)

    cmc = out["cmc"]
    depth = len(cmc)
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, depth + 1), cmc, marker="o", lw=2, color="#1f77b4")
    plt.xlabel("Rank")
    plt.ylabel("Identification accuracy")
    plt.title(f"CMC - {out['num_identities']} identities, {out['num_probe_images']} probes\n"
              f"Rank-1 = {cmc[0]:.4f}, Rank-5 = {out['rank5_accuracy']:.4f}")
    plt.ylim(0, 1.02)
    plt.xticks(range(1, depth + 1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(res / "cmc_curve.png", dpi=150)
    plt.close()

    (res / "gallery_probe_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"Rank-1 = {cmc[0]:.4f}   Rank-5 = {out['rank5_accuracy']:.4f}   "
          f"(chance {out['chance_rank1']:.4f})")

    if args.sweep:
        rows = []
        for k in range(1, args.sweep + 1):
            m, _ = identify(z_of, people, k, args.max_rank)
            m.pop("cmc")
            rows.append(m)
            print(f"  gallery={k}: Rank-1 {m['rank1_accuracy']:.4f}  "
                  f"Rank-5 {m['rank5_accuracy']:.4f}  "
                  f"mean rank {m['mean_rank_of_true_identity']:.2f}  "
                  f"({m['num_probe_images']} probes)")
        (res / "gallery_probe_multishot.json").write_text(json.dumps(rows, indent=2))

    if args.open_set:
        o = open_set(z_of, people, args.gallery_per_id, args.open_set, args.max_rank)
        (res / "open_set_metrics.json").write_text(json.dumps(o, indent=2))

        plt.figure(figsize=(7, 5))
        plt.semilogx(o["curve"]["fpir"], o["curve"]["dir"], lw=2, color="#1f77b4")
        for far, key in ((0.01, "dir_at_fpir_1pct"), (0.10, "dir_at_fpir_10pct")):
            plt.axvline(far, ls="--", color="grey", lw=1)
            plt.plot(far, o[key], "o", ms=7, label=f"DIR@FPIR={far:g} = {o[key]:.4f}")
        plt.xlabel("False Positive Identification Rate (unknowns wrongly accepted)")
        plt.ylabel("Detection & Identification Rate")
        plt.title(f"Open-set identification\n{o['enrolled_identities']} enrolled, "
                  f"{o['unknown_identities']} unknown identities")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.3, which="both")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(res / "open_set_curve.png", dpi=150)
        plt.close()

        print(f"open-set: {o['enrolled_identities']} enrolled / {o['unknown_identities']} unknown, "
              f"{o['known_probes']} known + {o['unknown_probes']} unknown probes")
        print(f"  DIR@FPIR=1%  : {o['dir_at_fpir_1pct']:.4f} (threshold {o['threshold_at_fpir_1pct']:.4f})")
        print(f"  DIR@FPIR=10% : {o['dir_at_fpir_10pct']:.4f} (threshold {o['threshold_at_fpir_10pct']:.4f})")


if __name__ == "__main__":
    main()
