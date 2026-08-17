"""Write a plain-text summary of every reported metric.

Reads the result JSONs and emits results/metrics_summary.txt, so the numbers in
the text file can never drift from the ones that were measured.

    python make_summary.py
"""

import argparse
import csv
import json
from pathlib import Path


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def rule(title, ch="="):
    return f"\n{ch * 78}\n{title}\n{ch * 78}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/metrics_summary.txt")
    args = ap.parse_args()

    root, res = Path(args.root), Path(args.results_dir)
    stats = load(root / "dataset_stats.json", {})
    hist = load(root / "checkpoints/training_history.json", {})
    ev = load(res / "evaluation_results.json", {})
    ex = load(res / "exhaustive_metrics.json", {})
    err = load(res / "error_summary.json", {})
    ms = load(res / "gallery_probe_multishot.json", [])
    op = load(res / "open_set_metrics.json", {})
    pairs = load(res / "pairs_test_stats.json", {})

    ver = ev.get("verification_test", {})
    val = ev.get("verification_validation", {})
    idn = ev.get("identification", {})
    thr = ev.get("threshold", {})
    L = []
    P = L.append

    P("FACE VERIFICATION / RE-IDENTIFICATION - METRICS SUMMARY")
    P("ResNet50 + ArcFace + batch-hard triplet, 512-D L2-normalised embeddings")
    P("All figures measured on identities never seen during training.")

    P(rule("1. DATASET"))
    sp = stats.get("splits", {})
    P(f"  Source              : {stats.get('dataset_name','-')}")
    P(f"  Downloaded from     : {stats.get('downloaded_from','-')}")
    P(f"  Licence             : {stats.get('licence','-')}")
    P(f"  Identities / images : {stats.get('num_identities','-')} / {stats.get('num_images','-')}")
    P(f"  Resolution          : {stats.get('image_size','-')} x {stats.get('image_size','-')} RGB")
    for k in ("train", "val", "test"):
        if k in sp:
            P(f"  {k:20s}: {sp[k]['identities']:>5} identities   {sp[k]['images']:>5} images")
    P("  Split policy        : identity-disjoint (no test person appears in training)")

    if hist.get("history"):
        best = max(hist["history"], key=lambda r: r["val_auc"])
        P(rule("2. TRAINING"))
        P(f"  Backbone            : {hist.get('backbone','resnet50')} @ {hist.get('size','-')}px")
        P(f"  Training set        : {hist.get('train_images','-')} images / "
          f"{hist.get('train_identities','-')} identities")
        P(f"  Epochs              : {len(hist['history'])}  (best epoch {best['epoch']} by val AUC)")
        P(f"  Validation AUC/EER  : {best['val_auc']:.4f} / {best['val_eer']*100:.2f}%")
        P(f"  Loss first -> last  : {hist['history'][0]['loss']:.2f} -> {hist['history'][-1]['loss']:.2f}")

    if pairs:
        P(rule("3. EVALUATION PAIRS"))
        P(f"  Protocol            : {pairs.get('split','test')} split, balanced sample")
        P(f"  Positive (genuine)  : {pairs['generated_positive']:,}")
        P(f"  Negative (impostor) : {pairs['generated_negative']:,}")
        P(f"  Duplicate pairs     : {pairs['duplicate_pairs']}")
        P(f"  Max possible genuine: {pairs['max_possible_positive']:,}")
        P(f"  Random seed         : {pairs['seed']}")

    if ver:
        P(rule("4. VERIFICATION  (1:1  -  are these two the same person?)"))
        P(f"  ROC-AUC             : {ver['roc_auc']:.4f}")
        P(f"  Equal Error Rate    : {ver['eer']*100:.2f}%   (at threshold {ver['eer_threshold']:.4f})")
        P(f"  Decision threshold  : {thr.get('value',0):.4f}")
        P(f"  Selected on         : {thr.get('selected_on','-')}")
        P(f"  Rule                : cosine >= threshold -> MATCH, else NON-MATCH")
        P(f"  Accuracy            : {ver['accuracy_at_chosen_threshold']*100:.2f}%")
        P(f"  Precision           : {ver['precision']:.4f}")
        P(f"  Recall              : {ver['recall']:.4f}")
        P(f"  F1 score            : {ver['f1']:.4f}")
        P("")
        P("  ERROR BREAKDOWN AT THE OPERATING THRESHOLD")
        P(f"    False accepts (FAR): {ver['false_accepts']:>5} / {ver['num_impostor']:,} impostor "
          f"pairs = {100*ver['false_accepts']/ver['num_impostor']:.2f}%")
        P(f"    False rejects (FRR): {ver['false_rejects']:>5} / {ver['num_genuine']:,} genuine  "
          f"pairs = {100*ver['false_rejects']/ver['num_genuine']:.2f}%")
        P("")
        P("  SCORE DISTRIBUTIONS")
        P(f"    Genuine  pairs     : mean {ver['genuine_mean']:+.4f}  std {ver['genuine_std']:.4f}")
        P(f"    Impostor pairs     : mean {ver['impostor_mean']:+.4f}  std {ver['impostor_std']:.4f}")
        P("")
        P("  TAR AT FIXED FAR  (5,000 impostor pairs - floor is FAR = 2.0e-04)")
        P(f"    TAR @ FAR = 1e-02  : {ver['tar_at_far_1pct']*100:6.2f}%   "
          f"(threshold {ver['threshold_at_far_1pct']:.4f})")
        P(f"    TAR @ FAR = 1e-03  : {ver['tar_at_far_0.1pct']*100:6.2f}%   "
          f"(threshold {ver['threshold_at_far_0.1pct']:.4f})")
        if val:
            P("")
            P(f"  Validation split reference: AUC {val['roc_auc']:.4f}, EER {val['eer']*100:.2f}%, "
              f"best accuracy {val['best_accuracy']*100:.2f}%")

    if ex:
        P(rule("5. EXHAUSTIVE VERIFICATION  (every pair, for low-FAR resolution)"))
        P(f"  Protocol            : {ex['protocol']}")
        P(f"  Images              : {ex['num_images']:,}")
        P(f"  Genuine pairs       : {ex['num_genuine']:,}")
        P(f"  Impostor pairs      : {ex['num_impostor']:,}")
        P(f"  Smallest FAR meas.  : {ex['smallest_measurable_far']:.2e}")
        P(f"  ROC-AUC / EER       : {ex['roc_auc']:.4f} / {ex['eer']*100:.2f}%")
        P("")
        P("       FAR        TAR      threshold   impostor pairs behind it")
        for k, v in ex["tar_at_far"].items():
            note = "" if v["impostor_pairs_at_this_far"] >= 100 else "   <- too few pairs to trust"
            P(f"    {float(k):9.0e}  {v['tar']*100:6.2f}%     {v['threshold']:.4f}     "
              f"{v['impostor_pairs_at_this_far']:>8.1f}{note}")

    if idn:
        P(rule("6. IDENTIFICATION  (1:N  -  who is this person?)"))
        P(f"  Gallery identities  : {idn['num_identities']}")
        P(f"  Enrolled per person : {idn['gallery_per_identity']}")
        P(f"  Gallery / probes    : {idn['num_gallery_images']} / {idn['num_probe_images']}")
        P(f"  Rank-1 accuracy     : {idn['rank1_accuracy']*100:.2f}%")
        P(f"  Rank-5 accuracy     : {idn['rank5_accuracy']*100:.2f}%")
        P(f"  Chance Rank-1       : {idn['chance_rank1']*100:.2f}%")
        P(f"  Mean rank of truth  : {idn['mean_rank_of_true_identity']:.2f}")
        if ms:
            P("")
            P("  EFFECT OF ENROLLING MORE IMAGES PER IDENTITY")
            P("    images   Rank-1    Rank-5   mean rank   probes")
            for r in ms:
                P(f"    {r['gallery_per_identity']:>6}  {r['rank1_accuracy']*100:6.2f}%  "
                  f"{r['rank5_accuracy']*100:6.2f}%   {r['mean_rank_of_true_identity']:>8.2f}   "
                  f"{r['num_probe_images']:>6}")

    if op:
        P(rule("7. OPEN-SET IDENTIFICATION  (some probes are not enrolled at all)"))
        P(f"  Enrolled identities : {op['enrolled_identities']}")
        P(f"  Unknown identities  : {op['unknown_identities']}")
        P(f"  Known / unknown probes: {op['known_probes']} / {op['unknown_probes']}")
        P(f"  Closed-set Rank-1   : {op['closed_set_rank1']*100:.2f}%  (same gallery)")
        P(f"  DIR @ FPIR = 1%     : {op['dir_at_fpir_1pct']*100:.2f}%   "
          f"(threshold {op['threshold_at_fpir_1pct']:.4f})")
        P(f"  DIR @ FPIR = 10%    : {op['dir_at_fpir_10pct']*100:.2f}%   "
          f"(threshold {op['threshold_at_fpir_10pct']:.4f})")
        P(f"  Mean top-1, known   : {op['known_top1_mean']:.4f}")
        P(f"  Mean top-1, unknown : {op['unknown_top1_mean']:.4f}")

    if err:
        P(rule("8. ERROR ANALYSIS  (every mistake at the operating threshold)"))
        P(f"  Threshold           : {err['threshold']:.4f}")
        P(f"  False accepts       : {err['false_accepts']} / {err['impostor_pairs']:,} impostor "
          f"pairs = {err['false_accept_rate']*100:.2f}% FAR")
        P(f"  False rejects       : {err['false_rejects']} / {err['genuine_pairs']:,} genuine  "
          f"pairs = {err['false_reject_rate']*100:.2f}% FRR")
        P(f"  Worst false accept  : {err['worst_false_accept']:.4f} (different people, scored high)")
        P(f"  Worst false reject  : {err['worst_false_reject']:.4f} (same person, scored low)")

        fa = res / "false_accepts.csv"
        if fa.exists():
            rows = list(csv.DictReader(fa.open(encoding="utf-8")))
            P("")
            P("  TOP 15 FALSE ACCEPTS  (different people the model scored as a match)")
            P(f"    {'score':>8}   {'margin':>7}   identity A  vs  identity B")
            for r in rows[:15]:
                P(f"    {float(r['cosine_similarity']):8.4f}   {float(r['margin']):+7.4f}   "
                  f"{r['name_a']}  vs  {r['name_b']}")
            P(f"    ... full list of {len(rows)} pairs in results/false_accepts.csv")

        fr = res / "false_rejects.csv"
        if fr.exists():
            rows = list(csv.DictReader(fr.open(encoding="utf-8")))
            P("")
            P("  TOP 15 FALSE REJECTS  (same person the model scored as a non-match)")
            P(f"    {'score':>8}   {'margin':>7}   identity")
            for r in rows[:15]:
                P(f"    {float(r['cosine_similarity']):8.4f}   {float(r['margin']):+7.4f}   "
                  f"{r['name_a']}")
            P(f"    ... full list of {len(rows)} pairs in results/false_rejects.csv")

    lm = load(res / "identity_audit.json")
    if lm:
        P(rule("9. IDENTITY LABEL AUDIT  (same name / different person, and vice versa)"))
        P(f"  Exact duplicate images        : {lm['exact_duplicate_images']}")
        P(f"  Flagged one-label-two-people  : {len(lm['suspect_one_label_two_people'])}")
        P(f"  Flagged two-labels-one-person : {len(lm['suspect_two_labels_one_person'])}")
        P("")
        P("  Pair generation trusts the folder label, so:")
        P("    one label / two people  -> genuine pairs become impostor pairs")
        P("    two labels / one person -> impostor pairs become genuine pairs")
        P("  Both push the measured numbers PESSIMISTIC. A label error cannot invent")
        P("  a correct match, so neither can flatter the model.")
        if lm["suspect_two_labels_one_person"]:
            P("")
            P("  closest distinct identities (look-alikes, not confirmed duplicates):")
            for c in lm["suspect_two_labels_one_person"][:6]:
                P(f"    {c['similarity']:.4f}  {c['name_a']}  vs  {c['name_b']}")

    ds = load(res / "dataset_roc_metrics.json")
    ml = load(res / "mlfw_metrics.json")
    if ds or ml:
        P(rule("10. CROSS-DATASET: LFW vs MLFW (masked faces)"))
        if ds:
            P("  Same checkpoint, same balanced pairing protocol, MLFW restricted to")
            P("  identities absent from the training split.")
            P("")
            P(f"    {'dataset':10s} {'pairs':>8s} {'AUC':>8s} {'EER':>8s} {'TAR@1%':>9s}")
            for k in ("LFW", "MLFW"):
                d = ds[k]
                P(f"    {k:10s} {d['pairs']:>8,} {d['roc_auc']:>8.4f} "
                  f"{d['eer']*100:>7.2f}% {d['tar_at_far_1pct']*100:>8.2f}%")
            c = ds["cost_of_mask"]
            P("")
            P(f"  Cost of the mask: -{c['auc_drop']:.4f} AUC, "
              f"+{c['eer_increase_pp']:.2f} pp EER, -{c['tar_at_far_1pct_drop_pp']:.2f} pp TAR@FAR=1%")
        if ml:
            P("")
            P("  MLFW official protocol, for reference:")
            P(f"    identities                  : {ml['identities']:,}")
            P(f"    of those seen in training   : {ml['identities_in_train']:,} "
              f"({ml['train_overlap_fraction']*100:.1f}%)")
            for key, lbl in [("official_(leaky)", "official (leaky)"),
                             ("leakage-free_subset", "leakage-free subset"),
                             ("leakage-free_balanced", "leakage-free balanced")]:
                if key in ml:
                    d = ml[key]
                    P(f"    {lbl:26s}: AUC {d['roc_auc']:.4f}  EER {d['eer']*100:5.2f}%  "
                      f"({d['pairs']:,} pairs)")
            P("")
            P("  MLFW is adversarial by construction: the same identity wears different")
            P("  masks while different identities wear the same mask, on top of CALFW's")
            P("  cross-age gap. This model saw no masked faces in training, so near-chance")
            P("  on the official protocol is the expected outcome rather than a defect.")

    P(rule("11. HOW TO READ THESE NUMBERS"))
    P("  Verification is 1:1 - compare two photos, clear a threshold. Identification")
    P("  is 1:N - beat every other enrolled person. Identification is harder, which is")
    P("  why Rank-1 is well below verification accuracy on the same embedding.")
    P("")
    P("  Raising the threshold lowers false accepts and raises false rejects; lowering")
    P("  it does the reverse. Both cannot fall together at a fixed embedding - only a")
    P("  better embedding lifts the whole ROC curve.")
    P("")
    P("  TAR at FAR = 1e-05 rests on about five impostor pairs even in the exhaustive")
    P("  protocol, so treat it as a plot endpoint rather than a reliable measurement.")
    P("")
    P("  Trained on ~7,000 images, orders of magnitude fewer than production systems.")
    P("  LFW skews towards light-skinned adult males in frontal, well-lit poses, so")
    P("  these figures are NOT evidence of fairness across demographics.")

    P(rule("FILES", "-"))
    for name, desc in [
        ("pairs_test.csv", "the 10,000 evaluation pairs (label 1 = genuine, 0 = impostor)"),
        ("pair_scores.csv", "every pair with its cosine similarity"),
        ("false_accepts.csv", "all impostor pairs accepted at the threshold"),
        ("false_rejects.csv", "all genuine pairs rejected at the threshold"),
        ("gallery_probe_results.csv", "per-probe identification result and rank of the true identity"),
        ("evaluation_results.json", "full verification + identification metrics"),
        ("exhaustive_metrics.json", "all-pairs low-FAR metrics"),
        ("roc_curve.png / roc_curve_exhaustive.png", "ROC curves"),
        ("similarity_distribution.png", "genuine vs impostor histograms"),
        ("threshold_analysis.png", "FAR / FRR / accuracy against threshold"),
        ("cmc_curve.png", "cumulative match characteristic"),
        ("open_set_curve.png", "detection & identification rate vs false alarms"),
        ("false_accepts_examples.png", "the worst false accepts, side by side"),
    ]:
        P(f"  {name:44s} {desc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
