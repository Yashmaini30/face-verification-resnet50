# Face Verification / Re-Identification

A ResNet50-based face verification and re-identification pipeline. Given two face
images it answers *"are these the same person?"* using cosine similarity between
512-D L2-normalised embeddings, and it also performs closed-set identification
against an enrolled gallery.

The ArcFace head and the batch-hard triplet loss are written from scratch. No
packaged face-recognition library (`face_recognition`, `deepface`, `insightface`,
`facenet-pytorch`) is used anywhere — the only pretrained component is the
ImageNet ResNet50 backbone.

---

## 1. Dataset

| | |
|---|---|
| **Dataset** | Labeled Faces in the Wild (LFW), *funneled* variant |
| **Canonical source** | `http://vis-www.cs.umass.edu/lfw/` — the dataset's original home at UMass Amherst. **This host stopped resolving in August 2026** (NXDOMAIN), so it is cited for attribution but is no longer reachable. |
| **Actually downloaded from** | <https://ndownloader.figshare.com/files/5976015> — the figshare mirror of `lfw-funneled.tgz` that `scikit-learn`'s `fetch_lfw_people` uses. `dataset_preparation.py` tries this first and falls back to UMass in case it returns. |
| **Licence / usage** | Free for **non-commercial research use**. The maintainers ask that it be used for research purposes and that the original technical report be cited. |
| **Provenance** | Collected by the LFW maintainers from **public news photographs** (Yahoo! News). No law-enforcement imagery, no mugshots, no criminal-record information. |
| **Citation** | G.B. Huang, M. Ramesh, T. Berg, E. Learned-Miller. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.* UMass Amherst TR 07-49, 2007. |

Nothing was scraped — the archive is fetched from the maintainers' own
distribution URL, and `dataset/` is a preprocessed derivative of it.

### After preprocessing

| Property | Value |
|---|---|
| Identities | **1,680** |
| Images | **9,164** |
| Resolution | 224 × 224 RGB |
| Face detection success | 9,116 / 9,164 (**99.5%**), 48 fell back to a central crop |
| Eye-alignment applied | 4,369 images |

Pipeline: Haar-cascade face detection → eye-line rotation when both eyes are
found → square crop with a 30% margin (reflect-padded near borders) → resize to
224×224 → ImageNet mean/std normalisation at load time.

### Splits — identity-disjoint

| Split | Identities | Images | Purpose |
|---|---|---|---|
| `train` | 1,500 | 7,096 | model fitting |
| `val` | 60 | 1,037 | checkpoint selection + threshold selection |
| `test` | 120 | 1,031 | final verification + gallery/probe metrics |

Splits are drawn over **identities, not images**. No person in `val` or `test`
appears anywhere in `train`, so every reported number measures generalisation to
unseen people. Identities need ≥ 4 images to be eligible for `val`/`test`; every
remaining identity with ≥ 2 images goes to training, which maximises training
classes without ever admitting an evaluation identity.

`dataset/` is not committed (145 MB, ~9k files) — regenerate it with step 1 below.

---

## 2. Installation

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

Tested on Python 3.12 (local, CPU) and Colab (Python 3.11, T4 GPU). Needs about
1 GB of disk for the LFW archive plus the preprocessed dataset.

---

## 3. Running the pipeline

### Step 1 — prepare the dataset (~10 min, downloads ~230 MB)

```bash
python dataset_preparation.py --raw-dir ../data_raw --out-dir dataset --size 224
```

Writes `dataset/person_XXX/img_YY.jpg`, `splits.json`, `dataset_stats.json` and
`identity_manifest.json` (which maps each `person_XXX` folder back to its
original LFW identity, for provenance).

### Step 2 — train

```bash
python train.py --root . --size 224 --epochs 30 --batch-p 16 --batch-k 4
```

Uses CUDA + mixed precision when a GPU is available, CPU otherwise. Writes
`checkpoints/best_model.pth` and `checkpoints/training_history.json`.

**On a free Colab GPU** — open `colab_train.ipynb`. Zip this folder, put
`submission.zip` on Google Drive, set the runtime to T4, run the cells, then copy
`best_model.pth` back into `checkpoints/`. Uploading the already-preprocessed
dataset (rather than re-preparing it on Colab) keeps both environments identical:
same images, same split, so the checkpoint matches the split the local evaluation
scripts use.

### Step 3 — evaluate

```bash
python generate_pairs.py --split test --n-positive 5000 --n-negative 5000 --out results/pairs_test.csv
python generate_pairs.py --split val  --n-positive 2500 --n-negative 2500 --out results/pairs_val.csv

python evaluate_pairs.py --pairs results/pairs_test.csv --out results/pair_scores.csv
python evaluate_pairs.py --pairs results/pairs_val.csv  --out results/pair_scores_val.csv

python gallery_probe.py --gallery-per-id 1
python roc_analysis.py --scores results/pair_scores.csv --val-scores results/pair_scores_val.csv
```

### Step 4 — try it on your own images (optional)

```bash
pip install fastapi uvicorn python-multipart
python demo_api.py --gallery test --gallery-per-id 2
```

Open <http://127.0.0.1:8000> for a browser form, or call the endpoints directly:

```bash
# same person?
curl -X POST http://127.0.0.1:8000/verify \
     -F "image_a=@photo1.jpg" -F "image_b=@photo2.jpg"
# {"cosine_similarity":0.4028,"threshold":0.1792,"match":true,"margin":0.2236}

# who is this, against the enrolled gallery?
curl -X POST "http://127.0.0.1:8000/identify?top_k=5" -F "image=@photo.jpg"
```

Uploads run through the same detect → align → crop → resize path as the training
data, so you can post an ordinary photograph rather than a pre-cropped face. The
threshold is read from `results/evaluation_results.json`, so the service always
decides with the same value the evaluation reported. If no face is detected the
request fails with 422 rather than silently scoring a centre crop.

A hosted Streamlit version of the same demo lives in [`demo/`](../demo/); it
pulls the weights from
[huggingface.co/yashMaini/face-verification-resnet50](https://huggingface.co/yashMaini/face-verification-resnet50)
rather than requiring a local checkpoint.

---

## 4. Model architecture

```
Face image (3 × 224 × 224)
        ↓
ResNet50 trunk  (torchvision, ImageNet-pretrained, classifier removed)
        ↓
Global average pooling → 2048-D
        ↓
Dropout(0.4) → Linear(2048 → 512, bias=False) → BatchNorm1d
        ↓
L2 normalisation
        ↓
512-D embedding on the unit hypersphere
```

Training-only heads, discarded at inference:

* `ArcFace` — additive angular margin classifier over the 1,500 training identities.
* `triplet_loss` — batch-hard mining on cosine distance.

At inference **only the L2-normalised embedding is used**. Classifier logits play
no part in any reported score — `evaluate_pairs.py` and `gallery_probe.py` never
even construct the ArcFace head.

---

## 5. Training methodology

**Objective:** `ArcFace cross-entropy + 1.0 × batch-hard triplet`, on the same
embeddings.

Why this combination:

1. **Why a classification loss at all.** With ~1,500 identities a classification
   objective gives a dense, stable gradient from every sample in the batch, and
   converges much faster and more reliably than a purely pairwise loss.
2. **Why ArcFace instead of plain softmax.** The system scores faces by cosine
   similarity. Plain cross-entropy optimises *unnormalised* inner products, so the
   geometry it learns is only loosely related to the test-time metric. ArcFace
   normalises both the embedding and the class weights and inserts an additive
   angular margin, so training optimises exactly the angular geometry that
   verification measures — and the margin forces an explicit gap between
   identities rather than merely separable ones.
3. **Why also a triplet term.** The classification head is thrown away at
   inference; it organises each identity around a learned class centre, not
   against the specific hard impostors the system actually meets. Batch-hard
   mining works directly on sample-to-sample cosine distance, pushing the nearest
   impostor away from the furthest genuine — precisely the quantity the ROC curve
   is built from.
4. **Why fine-tune from ImageNet.** ~7k training images is far too little to train
   a ResNet50 from scratch without severe overfitting. Pretrained low-level filters
   transfer well; a linear warm-up then cosine decay protects them during early
   epochs, and the randomly-initialised embedding and ArcFace heads get a 10×
   larger learning rate than the trunk.

**Batch construction.** A P×K sampler draws 16 identities × 4 images per batch.
Uniform random batches over 1,500 identities would almost never contain two
images of the same person, leaving the triplet term with no valid positives.

**Augmentation.** Horizontal flip, mild affine jitter (±8°, ±5% translate,
0.92–1.08 scale), colour jitter, light random erasing. Deliberately gentle —
faces are already cropped and aligned, and aggressive augmentation destroys
identity cues.

**Model selection.** The checkpoint with the highest verification AUC on the
validation identities is kept — the same protocol as the final test evaluation,
on people disjoint from both train and test.

---

## 6. Positive / negative pair generation

* **Positive pair** — two different images of the same identity → `label = 1`.
* **Negative pair** — one image each from two different identities → `label = 0`.
* **Balance** — 5,000 positive + 5,000 negative test pairs (2,500 + 2,500 for validation).

**Round-robin positive sampling.** Positives are drawn round-robin across
identities rather than uniformly over all within-identity combinations. LFW is
severely imbalanced — a single identity can contribute tens of thousands of the
possible genuine pairs — and uniform sampling would let a handful of people
dominate the genuine score distribution, giving an optimistic and
unrepresentative ROC. Round-robin gives every identity comparable weight until
its combinations run out.

**Duplicate prevention.** Every pair is keyed by its *sorted* `(path_a, path_b)`
tuple in a `seen` set shared by the positive and negative loops, so `(A,B)` and
`(B,A)` collapse to one key and no pair can be emitted twice. Positives come from
`itertools.combinations`, so no image is ever paired with itself; negatives draw
two *distinct* identities before drawing images.

**Leakage prevention.**

* Pairs come from a single split, and splits are identity-disjoint — so no test
  person, and no other photograph of a test person, was ever seen in training.
* The model plays no part in pair selection, so scores cannot bias which pairs
  get evaluated.
* The decision threshold is fitted on the **validation** pairs and applied
  unchanged to the test pairs.

---

## 7. Gallery / probe methodology

```
Gallery: person_001 → embedding
         person_002 → embedding
         ...
Probe image → ResNet50 → probe embedding
            → cosine similarity against every gallery template
            → highest similarity = predicted identity
```

* Each of the 120 test identities is enrolled with its **first image** (sorted by
  filename, so it is deterministic); the remaining images become probes. Gallery
  and probe images never overlap.
* With `--gallery-per-id > 1` the template is the L2-renormalised mean of that
  identity's enrolled embeddings.
* Both sides are unit-norm, so the probe × template matrix product *is* the
  cosine similarity matrix.
* Closed-set: every test identity is enrolled, so chance Rank-1 = 1/120 ≈ 0.83%.

Reports Rank-1, Rank-5, the full CMC curve, and a per-probe CSV including the
rank of the true identity for error analysis.

---

## 8. Results

ResNet50 trained for 30 epochs at 224×224 on a Colab T4 (41 s/epoch, ~21 minutes
total). Best epoch 22 by validation AUC. All numbers below are measured on the
**120 test identities, none of which appear in training**, and come from
`results/evaluation_results.json`.

Training loss fell 20.28 → 2.01 and train accuracy reached 89.2%. Validation AUC
rose to ~0.945 by epoch 10, then improved only slowly (0.9502 at epoch 13 →
0.9636 at epoch 22) while train accuracy climbed from 51% to 89% — the model
spends the back half of training fitting the training identities faster than it
gains transferable signal, and the final 8 epochs give nothing back. See
`results/training_curve.png`. The remaining headroom is in more data, not more
epochs.

**On input resolution.** An earlier run at 112×112 reached AUC 0.9539 and Rank-1
46.54%. The ResNet50 trunk is ImageNet-pretrained at 224×224 and the LFW source
images are 250×250, so the 112 crops were discarding detail the backbone was
built to use. Re-prepping at 224 cost 2.5× the training time and bought +0.016
AUC, −2.2 pp EER and **+9.3 pp Rank-1** — the largest single gain of any change
made to this pipeline.

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9700** |
| **Rank-1 accuracy** | **55.87%** |
| Rank-5 accuracy | 81.89% |
| **Final cosine threshold** | **0.1792** |
| Accuracy at threshold | 90.94% |
| EER | 8.98% |
| TAR @ FAR = 1% | 67.66% |
| TAR @ FAR = 0.1% | 37.60% |
| Precision / Recall / F1 | 0.9186 / 0.8984 / 0.9084 |
| False accepts | 398 / 5,000 impostor pairs |
| False rejects | 508 / 5,000 genuine pairs |
| Genuine similarity | 0.3984 ± 0.1751 |
| Impostor similarity | 0.0125 ± 0.1144 |

Validation split, for reference: AUC 0.9629, EER 9.86%, best accuracy 90.54% —
close enough to the test numbers that the threshold transfers cleanly.

**Identification.** 120 identities enrolled with one image each, 911 probes.
Rank-1 = 55.87% against a 0.83% chance baseline (67× better than random), and the
correct identity sits at mean rank 4.51 out of 120.

Single-image enrolment is the hardest setting, and it is the one reported above.
Enrolling more images per identity averages several embeddings into each
template, which cancels pose and lighting noise and lifts accuracy sharply — at
the cost of fewer images left to probe with:

| Images enrolled | Rank-1 | Rank-5 | Mean rank | Probes |
|---|---|---|---|---|
| 1 | 55.87% | 81.89% | 4.51 | 911 |
| 2 | 72.82% | 91.28% | 2.55 | 791 |
| 3 | 75.86% | 94.19% | 2.13 | 671 |

Reproduce with `python gallery_probe.py --gallery-per-id 1 --sweep 3`, which
writes `results/gallery_probe_multishot.json`. The headline figure stays at
single-image enrolment because it is the standard, hardest protocol — quoting
the 3-image number as the result would be picking the easiest setting.

### Open-set identification

Closed-set Rank-1 assumes every probe belongs to someone in the gallery. A real
system also has to say *"I don't know"*. Withholding 40 of the 120 test
identities from the gallery entirely and treating all 399 of their images as
impostor probes:

| Metric | Value |
|---|---|
| Enrolled / unknown identities | 80 / 40 |
| Known / unknown probes | 552 / 399 |
| Closed-set Rank-1 on this gallery | 64.13% |
| **DIR @ FPIR = 1%** | **20.11%** (threshold 0.4852) |
| DIR @ FPIR = 10% | 42.75% (threshold 0.3947) |
| Mean top-1 similarity, known / unknown | 0.4107 / 0.3175 |

A probe counts as correct only if its top-1 similarity clears the threshold
*and* points at the right person. Accuracy collapses from 64.1% to 20.1% once
the system must also reject strangers at a 1% false-alarm rate, and the last row
explains why: an unknown face scores 0.3175 against its nearest gallery entry
versus 0.4107 for a genuine match. Out of 80 enrolled people, the closest one to
a stranger is usually a plausible look-alike. Closing that gap needs a stronger
embedding, not a better threshold.

Reproduce with `python gallery_probe.py --gallery-per-id 1 --open-set 40`.

**Decision rule**

```
Threshold = 0.1792

Similarity >= 0.1792  ->  Match
Similarity <  0.1792  ->  Non-Match
```

The threshold is chosen on the validation split by maximising verification
accuracy, then applied unchanged to the test split. Picking it on the test scores
themselves would be a mild form of leakage and would overstate deployed accuracy.

### Evaluation integrity

`check_leakage.py` re-derives every anti-leakage claim from the files on disk
rather than trusting the code that produced them. It verifies identity-level
disjointness, image-level disjointness, that no two identities share an
identical image (all 9,164 images hash to 9,164 unique digests), that the pair
files contain no training identity and no duplicate or self-pairs, that labels
agree with folder identity, that gallery and probe sets never overlap, that
open-set unknowns are genuinely unenrolled and untrained, and that the threshold
and the checkpoint were both selected on validation data.

Two caveats that no such check can catch, stated explicitly:

**Operating points are read off the test curve.** TAR@FAR and DIR@FPIR pick
whichever threshold hits the target false-alarm rate *on the data being
measured*. This is the convention in the literature, but it is mildly
optimistic. Measuring the honest version — fit the threshold on validation, then
apply it unchanged to test — gives:

| Target FAR | Oracle TAR (test) | Threshold from val | Actual FAR | Actual TAR |
|---|---|---|---|---|
| 1% | 67.66% | 0.3149 | 0.86% | 66.30% |
| 0.1% | 37.60% | 0.4380 | 0.14% | 38.82% |

At FAR = 1% the threshold transfers almost perfectly — 0.5 points of optimism.
**At FAR = 0.1% the measurement is not trustworthy**, and the reason is sample
size rather than leakage: 0.1% of 2,500 validation impostor pairs is between two
and three pairs, so the threshold is being placed from a handful of points and
lands at 0.14% actual FAR. Treat TAR@FAR=0.1% as indicative only. The
headline numbers — AUC, EER, Rank-1 and the 0.1792 decision threshold — do not
depend on this and are unaffected.

**Relatives across splits.** LFW contains six members of the Bush family spread
across train, val and test. They are distinct people, so this is not leakage, but
related faces make impostor pairs *harder*. If anything it biases the reported
numbers down, not up.

### Exhaustive evaluation — resolving FAR below 10⁻⁴

The 5,000 + 5,000 sampled set the assessment asks for cannot resolve a FAR
smaller than 1/5000 = 2×10⁻⁴, so its low-FAR operating points rest on a handful
of pairs. Scoring **every** pair in the test split instead — all C(1031, 2) =
530,965 of them — raises the impostor count from 5,000 to **523,104** and makes
FAR down to 1.9×10⁻⁶ measurable.

```bash
python roc_analysis.py --exhaustive --split test
```

| | 5k/5k sample | **all pairs** |
|---|---|---|
| Genuine / impostor pairs | 5,000 / 5,000 | 7,861 / **523,104** |
| Smallest measurable FAR | 2.0×10⁻⁴ | **1.9×10⁻⁶** |
| ROC-AUC | 0.9700 | **0.9710** |
| EER | 8.98% | **8.92%** |
| TAR @ FAR = 10⁻² | 67.66% | **67.07%** (5,231 pairs) |
| TAR @ FAR = 10⁻³ | 37.60% *(5 pairs)* | **40.34%** (523 pairs) |
| TAR @ FAR = 10⁻⁴ | not measurable | **23.14%** (52 pairs) |
| TAR @ FAR = 10⁻⁵ | not measurable | **12.64%** (5 pairs) |

AUC and EER agree to within 0.001 and 0.06 points, which confirms the sampled
set was representative — the round-robin pair sampling did not bias the
distribution. What changes is *resolution*: TAR@FAR=10⁻³ moves from 37.60% to
40.34%, and the sampled figure was the noisy one, resting on about five pairs
against 523. `results/roc_curve_exhaustive.png` plots this down to 10⁻⁵, where
the curve is smooth rather than a staircase.

The headline numbers elsewhere in this README stay on the 5k/5k protocol, which
is what the assessment specifies; the exhaustive run is reported alongside it
because low-FAR claims need the extra pairs to mean anything.

### Attempts to improve low-FAR performance, and why they were rejected

TAR@FAR=0.1% (37.60%) is the weakest reported metric, so two standard families of
post-processing were tried. Both are fitted on the **training** split only
(identities disjoint from val and test), selected on **validation**, and only
then measured on **test**. Neither was adopted.

**1 — Embedding transforms.** Subtract the training-set mean embedding
(centering), optionally followed by ZCA whitening, before re-normalising.

| Variant | val TAR@1% / @0.1% | test TAR@1% / @0.1% | test AUC |
|---|---|---|---|
| baseline | 66.56% / 37.40% | **67.66% / 37.60%** | **0.9700** |
| centered | 65.24% / 37.36% | 66.76% / 36.20% | 0.9691 |
| centered + whitened | 56.32% / **41.24%** | 59.24% / 27.76% | 0.9390 |

Whitening gained **+3.8 points** of TAR@FAR=0.1% on validation and lost
**−9.8** on test, while dropping AUC by 0.031.

**2 — Cohort score normalisation.** Normalise each pair score against the
distribution of that image's similarities to a 2,000-image impostor cohort drawn
from the training split (Z-norm, symmetric S-norm, and adaptive S-norm over the
top-200 cohort scores).

| Variant | val TAR@1% / @0.1% | test TAR@1% / @0.1% | test AUC |
|---|---|---|---|
| baseline | 66.56% / 37.40% | **67.66% / 37.60%** | **0.9700** |
| Z-norm | 67.44% / 32.84% | 66.06% / 35.28% | 0.9693 |
| S-norm | 68.56% / 34.00% | 67.66% / 33.90% | 0.9699 |
| AS-norm (top-200) | **69.84% / 40.60%** | 68.02% / 35.90% | 0.9697 |

AS-norm gained **+3.2 points** on validation and lost **−1.7** on test.

**Conclusion.** Three separate validation-set gains failed to replicate on test,
which is the behaviour predicted above: at 5,000 impostor pairs, FAR = 0.1% is
about five pairs, so that operating point cannot support method selection. More
usefully, none of these transforms moved **AUC** — when post-processing cannot
extract additional signal, the constraint is the embedding itself rather than how
its scores are calibrated. That points back at the training set size (7,096
images), not at the scoring rule. Reaching TAR@FAR=0.1% of 60% would require a
corpus like CASIA-WebFace, not further tuning.

### How the threshold affects false accepts and false rejects

Cosine similarity rises with confidence that two faces match, and the threshold
is the single knob turning that score into a decision:

* **Raising the threshold** makes the system more suspicious. Fewer pairs clear
  the bar, so **false accepts (FAR) fall and false rejects (FRR) rise**. This is
  the right direction for access control, where wrongly admitting a stranger costs
  far more than asking a legitimate user to retry.
* **Lowering the threshold** does the reverse: more pairs are accepted, so **FAR
  rises and FRR falls** — trading security for convenience, which suits something
  like photo-album clustering.

The two error rates cannot be reduced at the same time for a fixed embedding; the
ROC curve *is* that trade-off swept over every threshold, and the EER is where
the two rates meet. Only a better embedding lifts the whole curve.
`results/threshold_analysis.png` plots FAR, FRR and accuracy against threshold.

---

## 9. Known limitations

* **Dataset scale.** ~7k training images across 1,500 identities is two to three
  orders of magnitude smaller than the corpora (MS1M, WebFace260M) behind
  production systems, so absolute accuracy sits below published state-of-the-art
  LFW figures, which train on large external datasets.
* **Demographic bias.** LFW comes from 2000s news photography and skews heavily
  towards light-skinned adult males in frontal, well-lit poses. Error rates on
  under-represented groups will be materially worse than the aggregate numbers
  here. **These results are not evidence of fairness across demographics.**
* **Detection and alignment.** The Haar cascade is fast and dependency-free but
  misses profile and heavily-occluded faces; 48 images fell back to a fixed
  central crop, which is safe on funnelled LFW imagery but would be unreliable in
  the wild. A modern detector with five-point landmark alignment (RetinaFace,
  MTCNN, YuNet) would improve crop consistency and typically several points of
  accuracy.
* **Open-set performance is weak.** It is measured above rather than assumed
  away, and rejecting strangers at a 1% false-alarm rate costs roughly three
  quarters of the closed-set accuracy. That is the number a deployment would
  actually live with, and it is the clearest evidence the embedding needs more
  training data before it could be used for anything real.
* **Single split, single seed.** Metrics come from one identity split and one
  training run; confidence intervals would need k-fold identity splits or
  repeated seeds.
* **TAR@FAR = 0.1% rests on too few pairs.** At 5,000 impostor pairs, a 0.1%
  false-accept rate is about five pairs, so that operating point carries wide
  uncertainty and cannot support method selection — three separate calibration
  techniques improved it on validation and lost on test. See *Evaluation
  integrity* above for the measurements.
* **No liveness / presentation-attack detection.** The system compares images,
  not live subjects — a printed photograph would verify as readily as a face.
* **Not a benchmark comparison.** These numbers use a custom identity-disjoint
  split, *not* the official LFW 6,000-pair protocol, so they are not directly
  comparable to published LFW accuracies.

---

## 10. Project structure

```
submission/
├── README.md
├── requirements.txt
├── dataset_preparation.py     # download, detect, align, crop, identity-disjoint split
├── model.py                   # ResNet50 embedding net, ArcFace head, triplet loss
├── train.py                   # training loop (CPU/GPU)
├── generate_pairs.py          # positive/negative pair generation
├── evaluate_pairs.py          # embeddings + cosine similarity
├── gallery_probe.py           # gallery/probe, Rank-1, Rank-5, CMC
├── roc_analysis.py            # ROC, AUC, EER, TAR@FAR, threshold selection
├── check_leakage.py           # independent leakage / duplication audit
├── demo_api.py                # optional FastAPI service to try it on your own images
├── colab_train.ipynb          # GPU training on Colab
├── checkpoints/
│   ├── best_model.pth
│   └── training_history.json
├── results/
│   ├── pairs_test.csv, pairs_val.csv
│   ├── pair_scores.csv, pair_scores_val.csv
│   ├── roc_curve.png
│   ├── similarity_distribution.png
│   ├── threshold_analysis.png
│   ├── cmc_curve.png
│   ├── training_curve.png
│   ├── open_set_curve.png
│   ├── evaluation_results.json
│   └── gallery_probe_results.csv
├── dataset/                   # person_001/img_01.jpg ... (not committed)
├── splits.json, dataset_stats.json, identity_manifest.json
└── report.pdf
```

## Reproducibility

Every stage is seeded (`--seed 42`): the identity split, pair sampling and
training initialisation. Re-running `dataset_preparation.py` on the same LFW
archive reproduces the same `person_XXX` folders and the same splits.
