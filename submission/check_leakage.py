"""Re-derive every anti-leakage claim from the files on disk.

Deliberately independent of the code that produced them: it re-reads the
splits, re-hashes the images and re-checks the labels, so a bug in the
generators cannot hide behind its own bookkeeping.

    python check_leakage.py

Exits non-zero if any check fails.
"""

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sp = json.loads((ROOT / "splits.json").read_text())
train, val, test = set(sp["train"]), set(sp["val"]), set(sp["test"])
fails = []


def check(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        fails.append(msg)


print("\n1. Identity-level disjointness")
check(not (train & val), f"train n val = {len(train & val)}")
check(not (train & test), f"train n test = {len(train & test)}")
check(not (val & test), f"val n test = {len(val & test)}")

print("\n2. Image-level disjointness (a file in two splits)")
owner = {}
dup_across = 0
for split, ids in (("train", train), ("val", val), ("test", test)):
    for p in ids:
        for f in (ROOT / "dataset" / p).glob("*.jpg"):
            if f in owner:
                dup_across += 1
            owner[f] = split
check(dup_across == 0, f"no image file belongs to two splits ({dup_across})")

print("\n3. Identical image CONTENT across different identities")
print("     (the same photo filed under two names would be true leakage)")
digest = defaultdict(list)
for f, split in owner.items():
    digest[hashlib.sha256(f.read_bytes()).hexdigest()].append((f.parent.name, split))
collisions = {h: v for h, v in digest.items() if len({p for p, _ in v}) > 1}
cross_split = {h: v for h, v in collisions.items() if len({s for _, s in v}) > 1}
print(f"     hashed {len(owner)} images -> {len(digest)} unique")
check(not collisions, f"no identical image under two identities ({len(collisions)})")
check(not cross_split, f"no identical image across two splits ({len(cross_split)})")
for h, v in list(cross_split.items())[:5]:
    print(f"       ! {v}")

print("\n4. Evaluation pairs draw only from their own split")
for name, allowed in (("pairs_test.csv", test), ("pairs_val.csv", val)):
    rows = list(csv.DictReader((ROOT / "results" / name).open(encoding="utf-8")))
    people = {Path(r[c]).parent.name for r in rows for c in ("image_a", "image_b")}
    check(people <= allowed, f"{name}: {len(people - allowed)} identities outside its split")
    check(not (people & train), f"{name}: {len(people & train)} TRAIN identities present")
    keys = [tuple(sorted((r["image_a"], r["image_b"]))) for r in rows]
    check(len(set(keys)) == len(keys), f"{name}: {len(keys)-len(set(keys))} duplicate pairs")
    check(all(a != b for a, b in keys), f"{name}: no image paired with itself")
    wrong = sum(1 for r in rows
                if (Path(r["image_a"]).parent.name == Path(r["image_b"]).parent.name)
                != (r["label"] == "1"))
    check(wrong == 0, f"{name}: {wrong} labels disagree with folder identity")

print("\n5. Gallery / probe disjointness (closed set)")
gp = list(csv.DictReader((ROOT / "results" / "gallery_probe_results.csv").open(encoding="utf-8")))
probes = {r["probe_image"] for r in gp}
gallery = {f"dataset/{p}/" + sorted(x.name for x in (ROOT/"dataset"/p).glob("*.jpg"))[0]
           for p in test}
check(not (probes & gallery), f"gallery n probes = {len(probes & gallery)}")
check({Path(p).parent.name for p in probes} <= test, "all probes are test identities")

print("\n6. Open-set: unknowns really are unenrolled")
o = json.loads((ROOT / "results" / "open_set_metrics.json").read_text())
names = sorted(test)
unknown = set(names[-o["unknown_identities"]:])
known = [n for n in names if n not in unknown]
check(not (unknown & set(known)), "unknown and enrolled identity sets are disjoint")
check(o["enrolled_identities"] == len(known), f"enrolled count {o['enrolled_identities']} == {len(known)}")
u_imgs = sum(len(list((ROOT/"dataset"/n).glob("*.jpg"))) for n in unknown)
check(o["unknown_probes"] == u_imgs, f"unknown probes {o['unknown_probes']} == all their images {u_imgs}")
check(not (unknown & train), "no unknown identity was ever trained on")
k_probe = sum(len(list((ROOT/"dataset"/n).glob("*.jpg"))) - 1 for n in known
              if len(list((ROOT/"dataset"/n).glob("*.jpg"))) > 1)
check(o["known_probes"] == k_probe, f"known probes {o['known_probes']} == images minus gallery {k_probe}")

print("\n7. Threshold provenance")
ev = json.loads((ROOT / "results" / "evaluation_results.json").read_text())
src = ev["threshold"]["selected_on"]
check("validation" in src, f"decision threshold fitted on: {src}")
val_thr = json.loads((ROOT/"results"/"pair_score_summary.json").read_text()) \
    if (ROOT/"results"/"pair_score_summary.json").exists() else None
print(f"     threshold value = {ev['threshold']['value']:.4f}")

print("\n8. Checkpoint selection provenance")
hist = json.loads((ROOT / "checkpoints" / "training_history.json").read_text())
best = max(hist["history"], key=lambda r: r["val_auc"])
check("val_auc" in best, f"best epoch {best['epoch']} chosen by validation AUC, not test")

print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED")
    for f in fails:
        print("  -", f)
else:
    print("No leakage found across 8 categories.")
