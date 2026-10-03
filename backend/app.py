"""Speech-emotion inference API (FastAPI + ONNX Runtime; no PyTorch, no librosa/numba -> small RAM footprint).

POST /predict  (multipart/form-data, field "file" = wav/flac/ogg/mp3 audio)
  -> {"emotion", "confidence", "probabilities", "windows": [...], "speech_detected", "latency_ms"}
GET  /health   -> {"status": "ok", ...}

Pre-processing is identical to the training notebook (`to_fixed`): 16 kHz mono -> librosa.effects.trim(top_db=30)
-> centre-crop / zero-pad to 3 s. Audio longer than 3 s is scored with a 3 s window / 1.5 s hop (50 % overlap) and the
window probabilities are averaged (confidence-weighted by speech energy gating).
"""
import io, json, os, time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "30"))       # reject / truncate very long uploads
MIN_RMS = float(os.environ.get("MIN_RMS", "0.003"))            # energy gate for "no speech"

meta = json.load(open(os.path.join(MODEL_DIR, "labels.json")))
LABELS, SR, N = meta["labels"], int(meta["sample_rate"]), int(meta["window_samples"])
TOP_DB = int(meta.get("top_db", 30))

so = ort.SessionOptions()
so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
SESSION = ort.InferenceSession(os.path.join(MODEL_DIR, MODEL_FILE), so, providers=["CPUExecutionProvider"])

# warm-up so the first real request is not slow
try:
    SESSION.run(None, {"waveform": np.zeros((1, N), np.float32)})
except Exception:
    pass

app = FastAPI(title="Speech Emotion Recognition API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def trim_silence(y: np.ndarray, top_db: float = TOP_DB, frame: int = 2048, hop: int = 512) -> np.ndarray:
    """NumPy re-implementation of librosa.effects.trim (ref=max) so training and serving use the same trimming."""
    if len(y) < frame:
        return y
    yp = np.pad(y, (frame // 2, frame // 2))                            # centre frames, zero padding
    n = 1 + (len(yp) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    power = np.mean(yp[idx] ** 2, axis=1)                                # rms**2 per frame
    db = 10 * np.log10(np.maximum(1e-10, power)) - 10 * np.log10(np.maximum(1e-10, power.max()))
    nz = np.flatnonzero(db > -top_db)
    if len(nz) == 0:
        return y[:0]
    return y[nz[0] * hop: min(len(y), (nz[-1] + 1) * hop)]


def to_fixed(y: np.ndarray) -> np.ndarray:
    """Same as the notebook: trim silence, then centre-crop / end-pad to exactly N samples."""
    if len(y) == 0:
        return np.zeros(N, np.float32)
    yt = trim_silence(y)
    if len(yt) < 0.1 * SR:
        yt = y
    if len(yt) >= N:
        s = (len(yt) - N) // 2
        yt = yt[s:s + N]
    else:
        yt = np.pad(yt, (0, N - len(yt)))
    return yt.astype(np.float32)


def load_audio(raw: bytes) -> np.ndarray:
    """Decode (wav/flac/ogg/mp3 via libsndfile) -> mono float32 @ 16 kHz (soxr HQ, same resampler family as librosa)."""
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True, frames=int(MAX_SECONDS * 48000))
    y = y.mean(axis=1)
    if sr != SR:
        y = soxr.resample(y, sr, SR, quality="HQ")
    return y[: int(MAX_SECONDS * SR)].astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def windows(y: np.ndarray):
    """Yield (start_sample, segment) with 50 % overlap; one window if the clip is <= 3 s."""
    if len(y) <= N:
        yield 0, y
        return
    hop = N // 2
    for s in range(0, len(y) - hop, hop):
        yield s, y[s:s + N]


@app.get("/")
def root():
    return {"service": "speech-emotion-recognition", "labels": LABELS, "architecture": meta.get("architecture"),
            "docs": "/docs", "predict": "POST /predict (multipart field 'file')"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_FILE, "labels": LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        y = load_audio(raw)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {type(e).__name__}") from e

    segs, wins = [], []
    for s, seg in windows(y):
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        if rms < MIN_RMS:                                    # energy gate: skip silent windows
            wins.append({"start_s": round(s / SR, 2), "speech": False})
            continue
        segs.append((len(wins), to_fixed(seg)))
        wins.append({"start_s": round(s / SR, 2), "speech": True})
    if not segs:
        return {"emotion": None, "confidence": 0.0, "probabilities": {}, "windows": wins, "speech_detected": False,
                "latency_ms": round((time.time() - t0) * 1000, 1)}

    probs = softmax(SESSION.run(None, {"waveform": np.stack([x for _, x in segs])})[0])    # (n_windows, n_classes)
    for (wi, _), p in zip(segs, probs):
        wins[wi].update(emotion=LABELS[int(p.argmax())], confidence=round(float(p.max()), 4))
    mean_p = probs.mean(axis=0)
    k = int(mean_p.argmax())
    return {"emotion": LABELS[k], "confidence": round(float(mean_p[k]), 4),
            "probabilities": {l: round(float(v), 4) for l, v in zip(LABELS, mean_p)},
            "windows": wins, "speech_detected": True, "latency_ms": round((time.time() - t0) * 1000, 1)}
