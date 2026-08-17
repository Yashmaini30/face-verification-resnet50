"""Face verification demo — Streamlit Community Cloud.

Self-contained: the model definition and preprocessing are inlined, and the
weights are pulled from the Hugging Face model repo at startup. Uploads go
through the same detect -> align -> crop -> resize path as the training data,
otherwise a raw photo would sit off the distribution the model was fitted on.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

HF_REPO = "yashMaini/face-verification-resnet50"
GITHUB = "https://github.com/Yashmaini30/face-verification-resnet50"
THRESHOLD = 0.1792          # fitted on the validation split
SIZE, MARGIN = 224, 0.30
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

st.set_page_config(page_title="Face Verification — ResNet50 + ArcFace",
                   page_icon="🧑‍🤝‍🧑", layout="centered")


class FaceNet(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        net = resnet50(weights=None)
        self.trunk = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(2048, dim, bias=False), nn.BatchNorm1d(dim)
        )

    def forward(self, x, normalize=True):
        z = self.head(self.trunk(x).flatten(1))
        return F.normalize(z) if normalize else z


class Cropper:
    def __init__(self):
        base = Path(cv2.data.haarcascades)
        self.face = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        self.eye = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))

    def align(self, img, gray, box):
        x, y, w, h = box
        roi = gray[y:y + int(0.6 * h), x:x + w]
        if roi.size == 0:
            return img
        eyes = self.eye.detectMultiScale(roi, 1.1, 6)
        if len(eyes) < 2:
            return img
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        pts = sorted([(x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes])
        (lx, ly), (rx, ry) = pts
        angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
        if abs(angle) < 1 or abs(angle) > 30:
            return img
        m = cv2.getRotationMatrix2D(((lx + rx) / 2, (ly + ry) / 2), angle, 1.0)
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = self.face.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(boxes) == 0:
            return None                      # no centre-crop fallback: be honest
        box = max(boxes, key=lambda b: b[2] * b[3])
        img = self.align(img, gray, box)

        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        half = max(w, h) * (1 + 2 * MARGIN) / 2
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = int(round(cx + half)), int(round(cy + half))
        l, t = max(0, -x0), max(0, -y0)
        r, b = max(0, x1 - img.shape[1]), max(0, y1 - img.shape[0])
        if l or t or r or b:
            img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REFLECT_101)
            x0, x1, y0, y1 = x0 + l, x1 + l, y0 + t, y1 + t
        face = img[y0:y1, x0:x1]
        if face.size == 0:
            return None
        return cv2.resize(face, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


@st.cache_resource(show_spinner="Loading model…")
def load():
    path = hf_hub_download(repo_id=HF_REPO, filename="best_model.pth")
    model = FaceNet()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    return model, Cropper(), tf


model, cropper, tf = load()


@torch.no_grad()
def embed(face_bgr):
    x = tf(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    z = model(x, normalize=False) + model(torch.flip(x, [3]), normalize=False)
    return F.normalize(z)[0].numpy()


def to_bgr(pil):
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def rank_probe(probe_face, gallery):
    """gallery: list of (name, face_bgr). Returns ranked (name, score, face)."""
    zp = embed(probe_face)
    scored = [(name, float(zp @ embed(face)), face) for name, face in gallery]
    return sorted(scored, key=lambda r: -r[1])


st.title("Face Verification — ResNet50 + ArcFace")
st.write(
    "Upload two photos and the model decides whether they show the same person. "
    "A ResNet50 fine-tuned with a hand-implemented ArcFace + batch-hard triplet "
    f"objective produces 512-D L2-normalised embeddings; the decision is their cosine "
    f"similarity against a threshold of **{THRESHOLD}** fitted on a validation split."
)
st.caption(
    "Measured on 120 LFW identities never seen in training — ROC-AUC 0.9700, "
    f"EER 8.98%, 90.94% verification accuracy.  [Code and full report]({GITHUB})"
)

tab_v, tab_i = st.tabs(["Verify — are these the same person?",
                        "Identify — who is this?"])

EX = Path(__file__).parent / "examples"
PRESETS = {
    "— upload my own —": None,
    "Same person (A)": ("annan_1.jpg", "annan_2.jpg"),
    "Same person (B)": ("karzai_1.jpg", "karzai_2.jpg"),
    "Different people (A)": ("annan_1.jpg", "ridge_1.jpg"),
    "Different people (B)": ("karzai_1.jpg", "ridge_1.jpg"),
}
with tab_v:
    choice = st.selectbox("Try an example, or upload your own photos", list(PRESETS))

    if PRESETS[choice]:
        a_img, b_img = (Image.open(EX / f) for f in PRESETS[choice])
    else:
        c1, c2 = st.columns(2)
        fa = c1.file_uploader("Image A", type=["jpg", "jpeg", "png"])
        fb = c2.file_uploader("Image B", type=["jpg", "jpeg", "png"])
        a_img = Image.open(fa) if fa else None
        b_img = Image.open(fb) if fb else None

    if a_img and b_img:
        fa_, fb_ = cropper(to_bgr(a_img)), cropper(to_bgr(b_img))
        if fa_ is None or fb_ is None:
            which = "first" if fa_ is None else "second"
            st.error(f"No face detected in the {which} image. Try a clearer, more frontal photo.")
        else:
            score = float(embed(fa_) @ embed(fb_))
            match = score >= THRESHOLD
            margin = score - THRESHOLD

            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(fa_, cv2.COLOR_BGR2RGB), caption="Detected face A")
            c2.image(cv2.cvtColor(fb_, cv2.COLOR_BGR2RGB), caption="Detected face B")

            if match:
                st.success(f"### SAME PERSON  ·  similarity {score:.4f}")
            else:
                st.error(f"### DIFFERENT PEOPLE  ·  similarity {score:.4f}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Cosine similarity", f"{score:.4f}")
            m2.metric("Threshold", f"{THRESHOLD}")
            m3.metric("Margin", f"{margin:+.4f}",
                      "clear" if abs(margin) > 0.15 else "borderline")
            st.progress(min(1.0, max(0.0, (score + 1) / 2)))
            st.caption(
                "Scores run from −1 to 1. On the held-out test set genuine pairs averaged "
                "**0.398** and impostor pairs **0.013**."
            )

with tab_i:
    st.write(
        "Enrol a few people by uploading one photo each — the filename becomes the label — "
        "then upload a probe photo. The probe is compared against every enrolled face by "
        "cosine similarity and ranked. This is the gallery/probe protocol from the report, "
        "run on your own images."
    )
    g_files = st.file_uploader("Gallery — one photo per person", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key="gal")
    p_file = st.file_uploader("Probe — who is this?", type=["jpg", "jpeg", "png"], key="probe")

    gallery, skipped = [], []
    for f in g_files or []:
        face = cropper(to_bgr(Image.open(f)))
        (gallery.append((Path(f.name).stem, face)) if face is not None else skipped.append(f.name))
    if skipped:
        st.warning("No face detected, skipped: " + ", ".join(skipped))
    if gallery:
        st.caption(f"{len(gallery)} identities enrolled")
        st.image([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in gallery],
                 caption=[n for n, _ in gallery], width=96)

    if p_file and len(gallery) >= 2:
        probe = cropper(to_bgr(Image.open(p_file)))
        if probe is None:
            st.error("No face detected in the probe image.")
        else:
            ranked = rank_probe(probe, gallery)
            c1, c2 = st.columns([1, 3])
            c1.image(cv2.cvtColor(probe, cv2.COLOR_BGR2RGB), caption="Probe", width=140)
            top_name, top_score, _ = ranked[0]
            gap = top_score - (ranked[1][1] if len(ranked) > 1 else -1.0)
            if top_score >= THRESHOLD:
                c2.success(f"### Rank-1: {top_name}\n\nsimilarity {top_score:.4f} "
                           f"· leads runner-up by {gap:+.4f}")
            else:
                c2.warning(f"### Closest: {top_name} ({top_score:.4f})\n\n"
                           f"below the {THRESHOLD} threshold — the probe may not be "
                           f"anyone in this gallery.")
            st.write("**Ranking**")
            for i, (name, score, face) in enumerate(ranked[:5], 1):
                a, b, c = st.columns([1, 3, 2])
                a.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), width=64)
                b.write(f"**{i}. {name}**")
                c.write(f"{score:.4f}")
            st.caption(
                "Closed-set identification always returns the nearest enrolled face, even for "
                "someone who is not enrolled. A large rank-1 to rank-2 gap is the confidence "
                "signal; a flat ranking means the model is unsure."
            )
    elif p_file:
        st.info("Enrol at least two people first, so there is something to rank against.")


with st.expander("Limitations"):
    st.markdown(
        "- Trained on ~7,000 images, orders of magnitude fewer than production systems, "
        "so expect errors on hard pairs — false-accept rate is about 8% at this threshold.\n"
        "- LFW skews towards light-skinned adult males in frontal, well-lit poses. Accuracy "
        "on under-represented groups will be worse, and these results are **not** evidence "
        "of fairness across demographics.\n"
        "- Face detection uses a Haar cascade, which misses profile and heavily-occluded "
        "faces.\n"
        "- Uploaded images are processed in memory for the comparison and are not stored."
    )
