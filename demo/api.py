"""Face verification REST API — deployable build.

Same model and preprocessing as the Streamlit demo, exposed as JSON endpoints.
Weights are pulled from the Hugging Face model repo at startup, so no checkpoint
lives in this repository.

    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints
    POST /verify    two images  -> cosine similarity + match decision
    POST /identify  gallery[] + probe -> ranked candidates
    GET  /health    liveness + model metadata
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# One thread keeps peak RSS down on small free-tier instances.
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))

HF_REPO = "yashMaini/face-verification-resnet50"
GITHUB = "https://github.com/Yashmaini30/face-verification-resnet50"
THRESHOLD = 0.1792
SIZE, MARGIN = 224, 0.30
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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


app = FastAPI(title="Face Verification API", version="1.0",
              description=f"ResNet50 + ArcFace face embeddings. Code: {GITHUB}")
state = {}


@app.on_event("startup")
def startup():
    path = hf_hub_download(repo_id=HF_REPO, filename="best_model.pth")
    model = FaceNet()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    state["model"] = model.eval()
    state["cropper"] = Cropper()
    state["tf"] = transforms.Compose([
        transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])


@torch.no_grad()
def embed_bytes(raw, label="image"):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"could not decode {label}")
    face = state["cropper"](img)
    if face is None:
        raise HTTPException(422, f"no face detected in {label}")
    x = state["tf"](Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    z = state["model"](x, normalize=False) + state["model"](torch.flip(x, [3]), normalize=False)
    return F.normalize(z)[0].numpy()


@app.post("/verify")
async def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    za = embed_bytes(await image_a.read(), "image_a")
    zb = embed_bytes(await image_b.read(), "image_b")
    score = float(za @ zb)
    return {"cosine_similarity": round(score, 4), "threshold": THRESHOLD,
            "match": bool(score >= THRESHOLD), "margin": round(score - THRESHOLD, 4)}


@app.post("/identify")
async def identify(probe: UploadFile = File(...), gallery: list[UploadFile] = File(...),
                   top_k: int = 5):
    if len(gallery) < 2:
        raise HTTPException(400, "upload at least two gallery images")
    zp = embed_bytes(await probe.read(), "probe")
    ranked = []
    for f in gallery:
        name = Path(f.filename or "unnamed").stem
        ranked.append({"identity": name,
                       "similarity": round(float(zp @ embed_bytes(await f.read(), name)), 4)})
    ranked.sort(key=lambda r: -r["similarity"])
    top = ranked[:top_k]
    return {"gallery_size": len(gallery), "threshold": THRESHOLD,
            "above_threshold": bool(top and top[0]["similarity"] >= THRESHOLD),
            "matches": top}


@app.get("/health")
async def health():
    return {"status": "ok", "model": HF_REPO, "threshold": THRESHOLD,
            "input_size": SIZE, "embedding_dim": 512, "loaded": "model" in state}


@app.get("/", response_class=HTMLResponse)
async def index():
    return f"""<!doctype html><meta charset=utf-8><title>Face Verification API</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:680px;margin:48px auto;padding:0 16px}}
code,pre{{background:#f4f7fa;border:1px solid #dce7ef;border-radius:6px}}
pre{{padding:12px;overflow-x:auto}} code{{padding:2px 5px}}</style>
<h1>Face Verification API</h1>
<p>ResNet50 fine-tuned with ArcFace + batch-hard triplet, producing 512-D
L2-normalised embeddings. Decisions are cosine similarity against a threshold of
<b>{THRESHOLD}</b> fitted on a validation split.</p>
<p>On 120 LFW identities never seen in training: ROC-AUC <b>0.9700</b>,
EER <b>8.98%</b>, verification accuracy <b>90.94%</b>.</p>
<h3>Endpoints</h3>
<pre>POST /verify     image_a, image_b            -> similarity + match
POST /identify   probe, gallery[], top_k     -> ranked candidates
GET  /health                                 -> liveness</pre>
<p>Interactive docs at <a href="/docs">/docs</a> &middot;
<a href="{GITHUB}">source and full report</a></p>
<p><small>Trained on ~7,000 images. LFW skews towards light-skinned adult males in
frontal poses, so this is not evidence of fairness across demographics. Images are
processed in memory and not stored.</small></p>"""
