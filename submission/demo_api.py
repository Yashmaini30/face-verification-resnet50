"""Small FastAPI service to try the trained model on your own images.

    pip install fastapi uvicorn python-multipart
    python demo_api.py

Then open http://127.0.0.1:8000 for a browser form, or POST to /verify and
/identify directly.

Uploads go through exactly the same detect -> align -> crop -> resize path as the
training data, otherwise a raw photo would sit off the distribution the model
was fitted on.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from dataset_preparation import Cropper
from model import embed, eval_tf, load_model

ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLD = 0.1792

app = FastAPI(title="Face Verification")
state = {}


def load_threshold():
    path = ROOT / "results" / "evaluation_results.json"
    if path.exists():
        return json.loads(path.read_text())["threshold"]["value"]
    return DEFAULT_THRESHOLD


def embed_upload(raw):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")

    face = state["cropper"].from_array(img, fallback=False)
    if face is None:
        raise HTTPException(422, "no face detected in image")

    from PIL import Image
    tensor = state["tf"](Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
    x = tensor.unsqueeze(0).to(state["device"])
    with torch.no_grad():
        z = state["model"](x, normalize=False) + state["model"](torch.flip(x, [3]), normalize=False)
    return F.normalize(z)[0].cpu().numpy()


@app.post("/verify")
async def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    za = embed_upload(await image_a.read())
    zb = embed_upload(await image_b.read())
    score = float(np.dot(za, zb))
    threshold = state["threshold"]
    return {
        "cosine_similarity": round(score, 4),
        "threshold": threshold,
        "match": bool(score >= threshold),
        "margin": round(score - threshold, 4),
    }


@app.post("/identify")
async def identify(image: UploadFile = File(...), top_k: int = 5):
    if state["templates"] is None:
        raise HTTPException(503, "no gallery enrolled; start with --gallery test")
    z = embed_upload(await image.read())
    sim = state["templates"] @ z
    order = np.argsort(-sim)[:top_k]
    return {
        "gallery_size": len(state["names"]),
        "matches": [{"identity": state["names"][i], "similarity": round(float(sim[i]), 4)}
                    for i in order],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "threshold": state["threshold"],
            "gallery_size": len(state["names"]) if state["templates"] is not None else 0}


PAGE = """
<!doctype html><meta charset=utf-8><title>Face Verification</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
 fieldset{border:1px solid #ccd;border-radius:8px;margin-bottom:24px;padding:16px}
 legend{font-weight:600;padding:0 6px}
 input[type=file]{display:block;margin:8px 0}
 button{padding:8px 18px;border:0;border-radius:6px;background:#1f6089;color:#fff;cursor:pointer}
 pre{background:#f4f7fa;border:1px solid #dce7ef;border-radius:6px;padding:12px;white-space:pre-wrap}
</style>
<h1>Face Verification</h1>
<p>Threshold <b>__THR__</b> &mdash; similarity at or above this is a match.</p>

<fieldset><legend>Verify &mdash; are these the same person?</legend>
<form onsubmit="send(event,'/verify','v')">
 <input type=file name=image_a accept=image/* required>
 <input type=file name=image_b accept=image/* required>
 <button>Compare</button>
</form><pre id=v></pre></fieldset>

<fieldset><legend>Identify &mdash; who is this?</legend>
<form onsubmit="send(event,'/identify','i')">
 <input type=file name=image accept=image/* required>
 <button>Search gallery</button>
</form><pre id=i></pre></fieldset>

<script>
async function send(e, url, out) {
  e.preventDefault();
  document.getElementById(out).textContent = 'working...';
  const r = await fetch(url, {method:'POST', body:new FormData(e.target)});
  document.getElementById(out).textContent = JSON.stringify(await r.json(), null, 2);
}
</script>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE.replace("__THR__", f"{state['threshold']:.4f}")


def setup(checkpoint, gallery_split, gallery_per_id, device):
    state["device"] = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state["model"] = load_model(checkpoint, state["device"])
    state["cropper"] = Cropper(size=state["model"].input_size)
    state["tf"] = eval_tf(state["model"].input_size)
    state["threshold"] = load_threshold()
    state["templates"] = None
    state["names"] = []

    if gallery_split:
        from dataset_preparation import load_split

        people = load_split(ROOT, gallery_split)
        names, paths, owner = sorted(people), [], []
        for person in names:
            chosen = people[person][:gallery_per_id]
            paths += chosen
            owner += [person] * len(chosen)

        z = embed(state["model"], paths, state["device"])
        templates = np.zeros((len(names), z.shape[1]), dtype=np.float32)
        row = {p: i for i, p in enumerate(names)}
        for vec, person in zip(z, owner):
            templates[row[person]] += vec
        templates /= np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12

        state["templates"] = templates
        state["names"] = names
        print(f"enrolled {len(names)} identities from the {gallery_split} split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--gallery", default="test", help="split to enrol, or empty for none")
    ap.add_argument("--gallery-per-id", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    setup(args.checkpoint, args.gallery, args.gallery_per_id, args.device)
    print(f"threshold {state['threshold']:.4f} - open http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
