"""Generate balanced positive/negative verification pairs from one split."""

import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path

from dataset_preparation import load_split


def make_pairs(people, n_pos, n_neg, seed=42):
    rng = random.Random(seed)
    names = sorted(people)
    seen = set()

    # LFW is very imbalanced, so positives are taken round-robin across
    # identities instead of uniformly over all within-identity combinations
    pools = {}
    for name in names:
        c = list(combinations(people[name], 2))
        rng.shuffle(c)
        pools[name] = c

    pos, done = [], set()
    while len(pos) < n_pos and len(done) < len(names):
        for name in names:
            if len(pos) >= n_pos:
                break
            if name in done:
                continue
            if not pools[name]:
                done.add(name)
                continue
            a, b = pools[name].pop()
            key = tuple(sorted((str(a), str(b))))
            if key not in seen:
                seen.add(key)
                pos.append((a, b, 1))

    neg, tries = [], 0
    while len(neg) < n_neg and tries < n_neg * 100:
        tries += 1
        na, nb = rng.sample(names, 2)
        a, b = rng.choice(people[na]), rng.choice(people[nb])
        key = tuple(sorted((str(a), str(b))))
        if key not in seen:
            seen.add(key)
            neg.append((a, b, 0))

    stats = {
        "requested_positive": n_pos,
        "requested_negative": n_neg,
        "generated_positive": len(pos),
        "generated_negative": len(neg),
        "num_identities": len(names),
        "num_images": sum(len(v) for v in people.values()),
        "max_possible_positive": sum(len(v) * (len(v) - 1) // 2 for v in people.values()),
        "duplicate_pairs": 0,
        "seed": seed,
    }
    return pos + neg, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-positive", type=int, default=5000)
    ap.add_argument("--n-negative", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/pairs_test.csv")
    args = ap.parse_args()

    people = {k: v for k, v in load_split(args.root, args.split).items() if v}
    pairs, stats = make_pairs(people, args.n_positive, args.n_negative, args.seed)
    stats["split"] = args.split

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "label"])
        for a, b, y in pairs:
            w.writerow([a.relative_to(args.root).as_posix(), b.relative_to(args.root).as_posix(), y])

    # plain-text mirror of the same pairs, LFW pairs.txt style: one pair per line
    txt = out.with_suffix(".txt")
    with txt.open("w", encoding="utf-8") as f:
        f.write(f"# {stats['generated_positive']} genuine + {stats['generated_negative']} "
                f"impostor pairs from the {args.split} split, seed {args.seed}\n")
        f.write("# label 1 = same person, 0 = different people\n")
        f.write("# image_a\timage_b\tlabel\n")
        for a, b, y in pairs:
            f.write(f"{a.relative_to(args.root).as_posix()}\t"
                    f"{b.relative_to(args.root).as_posix()}\t{y}\n")

    out.with_name(out.stem + "_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(pairs)} pairs -> {out} and {txt}")


if __name__ == "__main__":
    main()
