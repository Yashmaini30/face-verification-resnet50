"""Build report.pdf from the generated result files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

base = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=base["BodyText"], alignment=TA_JUSTIFY,
                      fontSize=9.5, leading=13.5, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=base["Heading1"], fontSize=16, spaceBefore=10,
                    spaceAfter=8, textColor=colors.HexColor("#12354f"))
H2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, spaceBefore=10,
                    spaceAfter=5, textColor=colors.HexColor("#1f6089"))
CAP = ParagraphStyle("cap", parent=base["BodyText"], fontSize=8,
                     textColor=colors.grey, spaceAfter=10)


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def pct(v):
    return f"{v * 100:.2f}%"


def table(rows):
    t = Table(rows, colWidths=(7.5 * cm, 8.5 * cm), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce7ef")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c8d4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def plot_training(history, path):
    epochs = [r["epoch"] for r in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(epochs, [r["loss"] for r in history], color="#d62728", lw=2, label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, [r["val_auc"] for r in history], color="#1f77b4", lw=2, label="Validation AUC")
    ax2.plot(epochs, [r["train_acc"] for r in history], color="#2ca02c", lw=2, ls="--",
             label="Train accuracy")
    ax2.set_ylabel("Validation AUC / train accuracy")
    ax2.set_ylim(0, 1.02)

    best = max(history, key=lambda r: r["val_auc"])
    ax2.plot(best["epoch"], best["val_auc"], "o", color="#1f77b4", ms=9)
    ax2.annotate(f"best AUC {best['val_auc']:.4f} (epoch {best['epoch']})",
                 (best["epoch"], best["val_auc"]), textcoords="offset points",
                 xytext=(-10, -18), ha="right", fontsize=8)

    lines = ax1.get_lines() + ax2.get_lines()[:2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    plt.title("Training progress")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure(name, caption, width=16 * cm):
    path = RESULTS / name
    if not path.exists():
        return []
    with PILImage.open(path) as im:
        ratio = im.height / im.width
    return [Image(str(path), width=width, height=width * ratio), Paragraph(caption, CAP)]


def main():
    stats = load(ROOT / "dataset_stats.json", {})
    ev = load(RESULTS / "evaluation_results.json", {})
    pairs = load(RESULTS / "pairs_test_stats.json", {})
    hist = load(ROOT / "checkpoints" / "training_history.json", {})

    ver = ev.get("verification_test", {})
    ident = ev.get("identification", {})
    thr = ev.get("threshold", {})

    doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Face Verification / Re-Identification")
    story = [
        Paragraph("Face Verification / Re-Identification", H1),
        Paragraph("ResNet50 embeddings, ArcFace + triplet training, cosine-similarity "
                  "verification and gallery/probe identification", BODY),
        Spacer(1, 6),
        Paragraph("1. Summary", H2),
    ]

    if ver:
        story.append(Paragraph(
            f"A ResNet50 backbone fine-tuned with a combined ArcFace + batch-hard-triplet "
            f"objective produces 512-D L2-normalised face embeddings. Evaluated on "
            f"<b>{ident.get('num_identities', '?')} identities never seen during training</b>, "
            f"cosine similarity gives <b>ROC-AUC = {ver['roc_auc']:.4f}</b>, "
            f"<b>EER = {pct(ver['eer'])}</b> and verification accuracy "
            f"<b>{pct(ver['accuracy_at_chosen_threshold'])}</b> at a threshold of "
            f"<b>{thr.get('value', 0):.4f}</b>. Closed-set identification reaches "
            f"<b>Rank-1 = {pct(ident.get('rank1_accuracy', 0))}</b> and "
            f"<b>Rank-5 = {pct(ident.get('rank5_accuracy', 0))}</b>.", BODY))
    else:
        story.append(Paragraph("Result files not found - run the evaluation scripts first.", BODY))

    story.append(Paragraph("2. Dataset", H2))
    if stats:
        sp = stats.get("splits", {})
        story += [
            Paragraph(f"<b>{stats['dataset_name']}</b> - {stats['source_url']}<br/>"
                      f"<i>Licence:</i> {stats['licence']} Images come from public news "
                      f"photographs; no law-enforcement or criminal-record imagery is involved.", BODY),
            table([
                ["Property", "Value"],
                ["Identities", str(stats["num_identities"])],
                ["Images", str(stats["num_images"])],
                ["Resolution", f"{stats['image_size']} x {stats['image_size']} RGB"],
                ["Train", f"{sp['train']['identities']} identities / {sp['train']['images']} images"],
                ["Validation", f"{sp['val']['identities']} identities / {sp['val']['images']} images"],
                ["Test", f"{sp['test']['identities']} identities / {sp['test']['images']} images"],
                ["Split policy", "Identity-disjoint"],
                ["Face detection", stats["preprocessing"]["detector"]],
                ["Alignment", stats["preprocessing"]["alignment"]],
            ]),
            Spacer(1, 8),
            Paragraph("Splits are drawn over <b>identities</b>, not images. Every person used for "
                      "validation or testing is absent from training, so the reported numbers "
                      "measure generalisation to unseen people.", BODY),
        ]

    story += [
        Paragraph("3. Model and training strategy", H2),
        Paragraph("<b>Architecture.</b> Face image (3x112x112) -> ResNet50 trunk "
                  "(ImageNet-pretrained, classifier removed) -> global average pooling (2048-D) -> "
                  "Dropout -> Linear(512, no bias) -> BatchNorm1d -> L2 normalisation -> 512-D "
                  "embedding on the unit hypersphere.", BODY),
        Paragraph("<b>Objective.</b> ArcFace additive-angular-margin cross-entropy over the "
                  "training identities plus a batch-hard triplet term on the same embeddings. "
                  "ArcFace was chosen over plain softmax because the system scores faces by cosine "
                  "similarity: normalising embeddings and class weights makes training optimise the "
                  "angular geometry that verification actually measures, and the margin forces an "
                  "explicit gap between identities. The triplet term is added because the "
                  "classifier head is discarded at inference - it organises samples around learned "
                  "class centres, whereas batch-hard mining works directly on the sample-to-sample "
                  "cosine distances the ROC is built from. Batches are drawn with a P x K sampler "
                  "so every batch has valid positives to mine.", BODY),
        Paragraph("<b>Transfer learning.</b> With only a few thousand training images a ResNet50 "
                  "trained from scratch overfits badly, so the trunk starts from ImageNet weights. "
                  "A linear warm-up followed by cosine decay protects them early on, and the "
                  "randomly-initialised embedding and ArcFace heads use a 10x larger learning "
                  "rate than the trunk.", BODY),
    ]

    if hist.get("history"):
        rows = hist["history"]
        best = max(rows, key=lambda r: r["val_auc"])
        story.append(table([
            ["Training detail", "Value"],
            ["Backbone", hist.get("backbone", "resnet50")],
            ["Training set", f"{hist.get('train_images', '?')} images / "
                             f"{hist.get('train_identities', '?')} identities"],
            ["Epochs", str(len(rows))],
            ["Best epoch (validation AUC)", str(best["epoch"])],
            ["Validation AUC", f"{best['val_auc']:.4f}"],
            ["Validation EER", pct(best["val_eer"])],
            ["Train accuracy at best epoch", pct(best["train_acc"])],
            ["Loss: first -> last epoch", f"{rows[0]['loss']:.2f} -> {rows[-1]['loss']:.2f}"],
            ["Time per epoch", f"{sum(r['seconds'] for r in rows) / len(rows):.0f} s (Colab T4)"],
            ["Selection criterion", "Highest verification AUC on validation identities"],
        ]))
        plot_training(rows, RESULTS / "training_curve.png")
        story += figure("training_curve.png",
                        "Figure 1 - Training loss against validation AUC. The gap between rising "
                        "train accuracy and a flattening validation AUC after ~epoch 20 is the "
                        "point where the model starts fitting the training identities rather than "
                        "learning transferable features.")

    story += [PageBreak(), Paragraph("4. Pair generation", H2)]
    if pairs:
        story += [table([
            ["Property", "Value"],
            ["Split", pairs.get("split", "test")],
            ["Positive pairs", str(pairs["generated_positive"])],
            ["Negative pairs", str(pairs["generated_negative"])],
            ["Identities", str(pairs["num_identities"])],
            ["Images", str(pairs["num_images"])],
            ["Max possible positives", str(pairs["max_possible_positive"])],
            ["Duplicate pairs", str(pairs["duplicate_pairs"])],
            ["Seed", str(pairs["seed"])],
        ]), Spacer(1, 8)]

    story.append(Paragraph(
        "Positive pairs are sampled <b>round-robin across identities</b> rather than uniformly "
        "over all within-identity combinations. LFW is severely imbalanced - a single identity can "
        "contribute tens of thousands of the possible genuine pairs - and uniform sampling would "
        "let a handful of people dominate the genuine score distribution, producing an optimistic "
        "and unrepresentative ROC. Negative pairs draw two distinct identities first, then one "
        "image from each. Every pair is keyed by its sorted path tuple in a hash set, so (A,B) and "
        "(B,A) collapse to one entry and no pair repeats; an image is never paired with itself. "
        "Pairs are drawn exclusively from the identity-disjoint test split and the model plays no "
        "part in selecting them, so neither training data nor model bias can leak into the "
        "evaluation.", BODY))

    story.append(Paragraph("5. Verification results", H2))
    if ver:
        story += [table([
            ["Metric", "Value"],
            ["ROC-AUC", f"{ver['roc_auc']:.4f}"],
            ["EER", pct(ver["eer"])],
            ["Selected threshold", f"{thr.get('value', 0):.4f}"],
            ["Threshold selected on", thr.get("selected_on", "-")],
            ["Accuracy at threshold", pct(ver["accuracy_at_chosen_threshold"])],
            ["Precision", f"{ver['precision']:.4f}"],
            ["Recall", f"{ver['recall']:.4f}"],
            ["F1 score", f"{ver['f1']:.4f}"],
            ["TAR @ FAR = 1%", pct(ver["tar_at_far_1pct"])],
            ["TAR @ FAR = 0.1%", pct(ver["tar_at_far_0.1pct"])],
            ["False accepts", f"{ver['false_accepts']} / {ver['num_impostor']}"],
            ["False rejects", f"{ver['false_rejects']} / {ver['num_genuine']}"],
            ["Genuine mean +/- std", f"{ver['genuine_mean']:.4f} +/- {ver['genuine_std']:.4f}"],
            ["Impostor mean +/- std", f"{ver['impostor_mean']:.4f} +/- {ver['impostor_std']:.4f}"],
        ]), Spacer(1, 8)]

    story += figure("roc_curve.png",
                    "Figure 2 - ROC curve (linear) and the low-FAR operating region (log FAR).")
    story += figure("similarity_distribution.png",
                    "Figure 3 - Genuine and impostor cosine-similarity distributions with the "
                    "decision threshold. The overlap is where verification errors occur.")

    story += [
        PageBreak(),
        Paragraph("6. Threshold analysis", H2),
        Paragraph(f"<b>Decision rule: cosine similarity &gt;= {thr.get('value', 0):.4f} -&gt; MATCH, "
                  f"otherwise NON-MATCH.</b> The threshold maximises verification accuracy on the "
                  f"validation split and is then applied unchanged to the test split. Selecting it "
                  f"on the test scores would be a mild form of leakage and would overstate deployed "
                  f"performance.", BODY),
        Paragraph("Raising the threshold makes the system more suspicious: fewer pairs clear the "
                  "bar, so false accepts fall while false rejects rise - the right operating point "
                  "for access control, where wrongly admitting a stranger costs far more than "
                  "asking a legitimate user to retry. Lowering it does the reverse, trading "
                  "security for convenience. The two error rates cannot both be reduced at a fixed "
                  "embedding; only a better embedding lifts the whole ROC curve.", BODY),
    ]
    story += figure("threshold_analysis.png",
                    "Figure 4 - FAR, FRR and accuracy across the threshold range. The FAR/FRR "
                    "crossing point is the EER.")

    story.append(Paragraph("7. Gallery / probe identification", H2))
    if ident:
        story += [
            Paragraph(f"Each of the {ident['num_identities']} test identities is enrolled with its "
                      f"first {ident['gallery_per_identity']} image(s); the remaining "
                      f"{ident['num_probe_images']} images act as probes. A probe embedding is "
                      f"compared against every gallery template by cosine similarity and assigned "
                      f"the highest-scoring identity.", BODY),
            table([
                ["Metric", "Value"],
                ["Gallery identities", str(ident["num_identities"])],
                ["Gallery images", str(ident["num_gallery_images"])],
                ["Probe images", str(ident["num_probe_images"])],
                ["Rank-1 accuracy", pct(ident["rank1_accuracy"])],
                ["Rank-5 accuracy", pct(ident["rank5_accuracy"])],
                ["Chance Rank-1", pct(ident["chance_rank1"])],
                ["Mean rank of true identity", f"{ident['mean_rank_of_true_identity']:.2f}"],
            ]),
            Spacer(1, 8),
        ]
    story += figure("cmc_curve.png",
                    "Figure 5 - CMC curve: probability the correct identity is within the top-k "
                    "gallery matches.")

    sweep = load(RESULTS / "gallery_probe_multishot.json")
    if sweep:
        story += [
            Paragraph("Single-image enrolment is the hardest setting. Enrolling more images per "
                      "identity averages several embeddings into each template, which cancels "
                      "pose and lighting noise and raises accuracy sharply - at the cost of "
                      "fewer probes left to test with.", BODY),
            table([["Images enrolled per identity", "Rank-1 / Rank-5 / mean rank / probes"]] +
                  [[str(r["gallery_per_identity"]),
                    f"{pct(r['rank1_accuracy'])} / {pct(r['rank5_accuracy'])} / "
                    f"{r['mean_rank_of_true_identity']:.2f} / {r['num_probe_images']}"]
                   for r in sweep]),
            Spacer(1, 8),
        ]

    op = load(RESULTS / "open_set_metrics.json")
    if op:
        story += [
            Paragraph("8. Open-set identification", H2),
            Paragraph(f"Closed-set Rank-1 assumes every probe belongs to someone in the gallery, "
                      f"so the system can always answer. A deployed system also has to be able to "
                      f"say \"I don't know\". Here {op['unknown_identities']} of the test identities "
                      f"are withheld from the gallery entirely and all "
                      f"{op['unknown_probes']} of their images become impostor probes, so accepting "
                      f"one is a false alarm rather than a ranking mistake. A probe is only counted "
                      f"correct if its top-1 similarity clears the threshold <i>and</i> points at "
                      f"the right person.", BODY),
            table([
                ["Metric", "Value"],
                ["Enrolled identities", str(op["enrolled_identities"])],
                ["Unknown identities", str(op["unknown_identities"])],
                ["Known / unknown probes", f"{op['known_probes']} / {op['unknown_probes']}"],
                ["Closed-set Rank-1 (same gallery)", pct(op["closed_set_rank1"])],
                ["DIR @ FPIR = 1%", pct(op["dir_at_fpir_1pct"])],
                ["Threshold @ FPIR = 1%", f"{op['threshold_at_fpir_1pct']:.4f}"],
                ["DIR @ FPIR = 10%", pct(op["dir_at_fpir_10pct"])],
                ["Threshold @ FPIR = 10%", f"{op['threshold_at_fpir_10pct']:.4f}"],
                ["Mean top-1 similarity, known", f"{op['known_top1_mean']:.4f}"],
                ["Mean top-1 similarity, unknown", f"{op['unknown_top1_mean']:.4f}"],
            ]),
            Spacer(1, 8),
            Paragraph(f"Performance drops sharply: {pct(op['closed_set_rank1'])} closed-set becomes "
                      f"{pct(op['dir_at_fpir_1pct'])} once the system must also reject strangers at "
                      f"a 1% false-alarm rate. The reason is visible in the last two rows - the mean "
                      f"top-1 score for an unknown face ({op['unknown_top1_mean']:.4f}) is not far "
                      f"below that of a genuine match ({op['known_top1_mean']:.4f}), because the "
                      f"nearest of 80 enrolled faces is often a plausible look-alike. Separating "
                      f"those needs a stronger embedding, not a better threshold.", BODY),
        ]
        story += figure("open_set_curve.png",
                        "Figure 6 - Detection & identification rate against the rate at which "
                        "unknown people are wrongly accepted.")

    story += [
        Paragraph("9. Known limitations", H2),
        Paragraph("<b>Dataset scale.</b> A few thousand training images across ~1.5k identities is "
                  "two to three orders of magnitude smaller than the corpora (MS1M, WebFace260M) "
                  "behind production face recognition, so absolute accuracy sits below published "
                  "state-of-the-art LFW figures, which train on large external datasets.", BODY),
        Paragraph("<b>Demographic bias.</b> LFW comes from 2000s news photography and skews heavily "
                  "towards light-skinned adult males in frontal, well-lit poses. Error rates on "
                  "under-represented groups will be materially worse than the aggregate numbers "
                  "here, which are not evidence of fairness across demographics.", BODY),
        Paragraph("<b>Detection and alignment.</b> The Haar cascade is fast and dependency-free but "
                  "misses profile and heavily-occluded faces; those images fall back to a fixed "
                  "central crop, safe on funnelled LFW imagery but unreliable in the wild. A modern "
                  "detector with five-point landmark alignment (RetinaFace, MTCNN, YuNet) would "
                  "improve crop consistency and typically several points of accuracy.", BODY),
        Paragraph("<b>Open-set performance is weak.</b> Section 8 measures it rather than assuming "
                  "it away, and the answer is that rejecting strangers at a 1% false-alarm rate "
                  "costs roughly three quarters of the closed-set accuracy. That is the number a "
                  "deployment would actually live with, and it is the clearest evidence that the "
                  "embedding needs more training data before it could be used for anything real.", BODY),
        Paragraph("<b>Single split, single seed.</b> Metrics come from one identity split and one "
                  "training run; confidence intervals would need k-fold splits or repeated seeds.", BODY),
        Paragraph("<b>No liveness detection.</b> The system compares images, not live subjects - a "
                  "printed photograph would verify as readily as a real face.", BODY),
        Paragraph("<b>Not a benchmark comparison.</b> These numbers use a custom identity-disjoint "
                  "split, not the official LFW 6,000-pair protocol, so they are not directly "
                  "comparable to published LFW accuracies.", BODY),
    ]

    doc.build(story)
    print(f"wrote {ROOT / 'report.pdf'}")


if __name__ == "__main__":
    main()
