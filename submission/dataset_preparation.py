"""Build the face dataset from LFW: detect, align, crop, resize, split."""

import argparse
import json
import random
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# The canonical UMass host (vis-www.cs.umass.edu) stopped resolving in Aug 2026,
# so the figshare mirror that scikit-learn uses is tried first and UMass is kept
# as a fallback in case it comes back.
URLS = [
    "https://ndownloader.figshare.com/files/5976015",
    "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
]

SOURCES = {
    "lfw_funneled": ("funneled", "https://ndownloader.figshare.com/files/5976015"),
    "lfw-deepfunneled": ("deep-funneled", "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"),
    "lfw": ("original (unfunneled)", "http://vis-www.cs.umass.edu/lfw/lfw.tgz"),
}

MIN_EVAL = 4
MIN_TRAIN = 2
SIZE = 224
MARGIN = 0.30
SEED = 42


def load_split(root, split):
    splits = json.loads((Path(root) / "splits.json").read_text())
    base = Path(root) / "dataset"
    return {p: sorted((base / p).glob("*.jpg")) for p in sorted(splits[split])}


def fetch(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in ("lfw-deepfunneled", "lfw_funneled", "lfw"):
        if (raw_dir / name).is_dir():
            return raw_dir / name

    tgz = raw_dir / "lfw.tgz"
    if not tgz.exists():
        for url in URLS:
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, tgz)
                break
            except Exception as e:
                print(f"  failed: {e}")
        else:
            raise RuntimeError("could not download LFW")

    print("extracting")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(raw_dir)

    for name in ("lfw-deepfunneled", "lfw_funneled", "lfw"):
        if (raw_dir / name).is_dir():
            return raw_dir / name
    raise RuntimeError("no LFW folder after extraction")


# ArcFace's canonical 5-point template, defined at 112x112 and scaled to `size`.
# Aligning every face onto these coordinates removes in-plane rotation, scale and
# translation, so the network sees eyes, nose and mouth in the same place every time.
TEMPLATE_112 = np.float32([
    [38.2946, 51.6963],   # right eye
    [73.5318, 51.5014],   # left eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # right mouth corner
    [70.7299, 92.2041],   # left mouth corner
])

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")


def yunet_model(path="../models/yunet.onnx"):
    """Fetch the YuNet detector weights once, then load them."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading YuNet detector -> {path}")
        urllib.request.urlretrieve(YUNET_URL, path)
    return str(path)


class Cropper:
    """YuNet detection + 5-point landmark alignment onto the ArcFace template.

    Falls back to the Haar cascade when YuNet finds nothing, so a detector miss
    degrades to the previous behaviour instead of dropping the image.
    """

    def __init__(self, size=SIZE, margin=MARGIN, model=None):
        cascades = Path(cv2.data.haarcascades)
        self.face = cv2.CascadeClassifier(str(cascades / "haarcascade_frontalface_default.xml"))
        self.eye = cv2.CascadeClassifier(str(cascades / "haarcascade_eye.xml"))
        if self.face.empty():
            raise RuntimeError("could not load haar cascade")
        self.size = size
        self.margin = margin
        self.stats = defaultdict(int)

        self.template = TEMPLATE_112 * (size / 112.0)
        try:
            self.yunet = cv2.FaceDetectorYN.create(yunet_model(model or "../models/yunet.onnx"),
                                                   "", (320, 320), 0.7, 0.3, 5000)
        except Exception as e:                       # keep working without the detector
            print(f"YuNet unavailable ({e}); falling back to Haar + eye alignment")
            self.yunet = None

    def landmarks(self, img):
        """Return the 5 landmark points of the most confident face, or None."""
        if self.yunet is None:
            return None
        h, w = img.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[14])          # highest detection score
        return np.float32(f[4:14]).reshape(5, 2)

    def align_landmarks(self, img, pts):
        """Similarity transform mapping the 5 points onto the canonical template."""
        m, _ = cv2.estimateAffinePartial2D(pts, self.template, method=cv2.LMEDS)
        if m is None:
            return None
        return cv2.warpAffine(img, m, (self.size, self.size), flags=cv2.INTER_LINEAR)

    def detect(self, gray):
        boxes = self.face.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(boxes) == 0:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])

    def align(self, img, gray, box):
        x, y, w, h = box
        roi = gray[y:y + int(0.6 * h), x:x + w]
        if roi.size == 0:
            return img
        eyes = self.eye.detectMultiScale(roi, 1.1, 6)
        if len(eyes) < 2:
            self.stats["not_aligned"] += 1
            return img

        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        pts = sorted([(x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes])
        (lx, ly), (rx, ry) = pts
        angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
        if abs(angle) < 1 or abs(angle) > 30:
            self.stats["not_aligned"] += 1
            return img

        center = ((lx + rx) / 2, (ly + ry) / 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.stats["aligned"] += 1
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def crop(self, img, box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        half = max(w, h) * (1 + 2 * self.margin) / 2
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = int(round(cx + half)), int(round(cy + half))

        # pad instead of clipping so faces near the border stay centred
        l, t = max(0, -x0), max(0, -y0)
        r, b = max(0, x1 - img.shape[1]), max(0, y1 - img.shape[0])
        if l or t or r or b:
            img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REFLECT_101)
            x0, x1, y0, y1 = x0 + l, x1 + l, y0 + t, y1 + t
        return img[y0:y1, x0:x1]

    def __call__(self, path):
        img = cv2.imread(str(path))
        if img is None:
            self.stats["unreadable"] += 1
            return None
        return self.from_array(img)

    def from_array(self, img, fallback=True):
        """Crop and align the face. With fallback=False, give up if nothing is found.

        Preferred path is YuNet landmarks + similarity transform. If the detector
        misses, drop back to Haar detection with eye-line rotation; if that misses
        too, a centre crop (safe on funnelled LFW, wrong for arbitrary uploads,
        hence the fallback flag).
        """
        pts = self.landmarks(img)
        if pts is not None:
            face = self.align_landmarks(img, pts)
            if face is not None:
                self.stats["landmark_aligned"] += 1
                return face
            self.stats["align_failed"] += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = self.detect(gray)
        if box is None:
            self.stats["detect_fail"] += 1
            if not fallback:
                return None
            side = int(0.5 * min(img.shape[:2]))
            box = (img.shape[1] // 2 - side // 2, img.shape[0] // 2 - side // 2, side, side)
        else:
            self.stats["detect_ok"] += 1
            img = self.align(img, gray, box)

        face = self.crop(img, box)
        if face.size == 0:
            self.stats["empty_crop"] += 1
            return None
        return cv2.resize(face, (self.size, self.size), interpolation=cv2.INTER_AREA)


def make_splits(people, n_test, n_val, seed):
    eligible = sorted(k for k, v in people.items() if len(v) >= MIN_EVAL)
    random.Random(seed).shuffle(eligible)
    if len(eligible) < n_test + n_val:
        raise RuntimeError(f"only {len(eligible)} identities have >= {MIN_EVAL} images")

    test = sorted(eligible[:n_test])
    val = sorted(eligible[n_test:n_test + n_val])
    held = set(test) | set(val)
    train = sorted(k for k, v in people.items() if k not in held and len(v) >= MIN_TRAIN)
    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="../data_raw")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    lfw = fetch(args.raw_dir)
    people = {}
    for d in sorted(p for p in lfw.iterdir() if p.is_dir()):
        imgs = sorted(d.glob("*.jpg"))
        if imgs:
            people[d.name] = imgs
    print(f"raw: {len(people)} identities, {sum(len(v) for v in people.values())} images")

    splits = make_splits(people, args.n_test, args.n_val, args.seed)
    for k, v in splits.items():
        print(f"  {k}: {len(v)} identities")

    cropper = Cropper(size=args.size)
    ordered = sorted((s, p) for s in splits for p in splits[s])
    ordered.sort(key=lambda t: t[1])

    manifest, folders, total = {}, {"train": [], "val": [], "test": []}, 0
    landmarks = {}
    for i, (split, original) in enumerate(ordered, 1):
        folder = f"person_{i:03d}"
        dest = out / folder
        dest.mkdir(parents=True, exist_ok=True)

        n = 0
        for src in people[original]:
            raw = cv2.imread(str(src))
            if raw is None:
                cropper.stats["unreadable"] += 1
                continue
            pts = cropper.landmarks(raw)
            face = cropper.from_array(raw)
            if face is None:
                continue
            n += 1
            name = f"img_{n:02d}.jpg"
            cv2.imwrite(str(dest / name), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if pts is not None:
                # landmarks in the ORIGINAL image frame, so the crop is reproducible
                landmarks[f"{folder}/{name}"] = {
                    "source": str(src.relative_to(lfw.parent)).replace("\\", "/"),
                    "right_eye": [round(float(pts[0][0]), 2), round(float(pts[0][1]), 2)],
                    "left_eye": [round(float(pts[1][0]), 2), round(float(pts[1][1]), 2)],
                    "nose": [round(float(pts[2][0]), 2), round(float(pts[2][1]), 2)],
                    "mouth_right": [round(float(pts[3][0]), 2), round(float(pts[3][1]), 2)],
                    "mouth_left": [round(float(pts[4][0]), 2), round(float(pts[4][1]), 2)],
                }

        if n < (MIN_EVAL if split in ("val", "test") else MIN_TRAIN):
            shutil.rmtree(dest)
            cropper.stats["dropped"] += 1
            continue

        manifest[folder] = {"original_identity": original, "split": split, "num_images": n}
        folders[split].append(folder)
        total += n
        if i % 200 == 0:
            print(f"  {i}/{len(ordered)}")

    variant, download_url = SOURCES.get(lfw.name, (lfw.name, "unknown"))
    stats = {
        "dataset_name": f"Labeled Faces in the Wild ({variant})",
        "archive_folder": lfw.name,
        "downloaded_from": download_url,
        "canonical_source": "http://vis-www.cs.umass.edu/lfw/ (offline since Aug 2026)",
        "licence": "Free for non-commercial research use; images from public news photographs.",
        "image_size": args.size,
        "seed": args.seed,
        "num_identities": len(manifest),
        "num_images": total,
        "splits": {k: {"identities": len(v), "images": sum(manifest[f]["num_images"] for f in v)}
                   for k, v in folders.items()},
        "preprocessing": {
            "detector": "OpenCV YuNet (5-point landmarks), Haar cascade fallback",
            "alignment": "similarity transform of the 5 landmarks onto the ArcFace template",
            "crop_margin": MARGIN,
            "fallback": "central 50% box when detection fails",
        },
        "detection_stats": dict(cropper.stats),
    }

    Path("landmarks.json").write_text(json.dumps(landmarks, indent=1))
    Path("dataset_stats.json").write_text(json.dumps(stats, indent=2))
    Path("splits.json").write_text(json.dumps(folders, indent=2))
    Path("identity_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
