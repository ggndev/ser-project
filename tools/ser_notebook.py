# %% [markdown]
# # Speech Emotion Recognition (SER) — Two-Protocol Evaluation
#
# **Stack:** PyTorch · torchaudio · 🤗 Transformers (`facebook/wav2vec2-base`) · librosa · silero-VAD · ONNX Runtime
# **Runtime:** **Kaggle Notebook** (free GPU, Accelerator ▸ GPU T4 ×2, *Internet: On*) — also runs unchanged on Google Colab (T4).
#
# ## What this notebook does
# | Step | Content |
# |---|---|
# | 1 | Parse **4 datasets** (TESS, RAVDESS, SAVEE, CREMA-D), harmonise to 7 emotions, extract **speaker IDs** |
# | 2 | Pre-process: 16 kHz mono → trim silence → fixed 3 s window; log-mel + Wav2Vec2 feature paths |
# | 3 | Augmentation (noise, pitch ±2 st, time-stretch 0.9–1.1×, random gain) — **train split only, applied after splitting** |
# | 4 | Model A: CNN + BiLSTM + Attention on log-mel · Model B: fine-tuned Wav2Vec2-base |
# | 5 | **Protocol 1** stratified random 80/10/10 split **and** **Protocol 2** 5-fold speaker-independent GroupKFold — reported side by side, never conflated |
# | 6 | Real-time demo: silero-VAD + 3 s sliding window (50 % overlap) + ONNX/TorchScript export |
# | 7 | Final report generated **from the measured numbers** (nothing is hard-coded) |
#
# ## How to run
# **Kaggle:** *Add Input* ▸ search & add the 4 datasets (`ravdess-emotional-speech-audio`, `cremad`, `surrey-audiovisual-expressed-emotion-savee`, `toronto-emotional-speech-set-tess`), turn **Internet On** (needed to download `facebook/wav2vec2-base` and `silero-vad`), pick **GPU T4 ×2** (one GPU is used).
# **Colab:** upload your `kaggle.json`; the data is downloaded automatically with `opendatasets`.
#
# 1. Run top to bottom. Long stages are **resumable**: results are written to `WORK_DIR/results_*.json` after each model/fold; finished stages are skipped on re-run. On Kaggle use *Save Version ▸ Save & Run All (Commit)* for an unattended run — outputs persist in the *Output* tab.
# 2. Set `QUICK_RUN = True` in the config cell for a ~10 min smoke test on a small subset.
# 3. **Staged workflow (saves GPU hours):** run through *Protocol 1 – Wav2Vec2* first (≈1 h). Only if the random-split accuracy looks right, run Protocol 2 + export.
#
# > **Honesty note.** All accuracy numbers in the final summary are computed by the code below. Protocol 1 and Protocol 2 answer *different questions* (see the markdown before the comparison table), and are always labelled.

# %% [markdown]
# ## 0. Install requirements

# %%
# Colab/Kaggle already ship torch / torchaudio / numpy / pandas / sklearn / transformers. We add the rest (needs Internet On).
# (webrtc-free: silero-vad is used for voice-activity detection.)
!pip -q install librosa soundfile opendatasets "transformers>=4.40" accelerate onnx onnxscript onnxruntime silero-vad seaborn joblib

# %% [markdown]
# ## 0b. Configuration & imports
# All knobs live in this one cell. Defaults are sized for a T4 and total roughly 3–4 h end-to-end; reduce `P2_EPOCHS_W2V` / set `RUN_P2_W2V=False` if you are short on time.

# %%
import os, sys, json, math, random, time, glob, re, copy, warnings
import numpy as np
import pandas as pd
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import librosa, soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings("ignore")

# ---------------- configuration ----------------
QUICK_RUN   = False     # True -> tiny subset + 1-2 epochs (pipeline smoke test)
USE_DRIVE   = False     # True -> keep WORK_DIR on Google Drive so progress survives disconnects
SEED        = 42
SR          = 16_000                    # target sample rate
DUR         = 3.0                       # fixed window (seconds)
N_SAMPLES   = int(SR * DUR)             # 48 000 samples per clip
TOP_DB      = 30                        # silence threshold for librosa.effects.trim
EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID    = {e: i for i, e in enumerate(EMOTIONS)}
N_CLASSES   = len(EMOTIONS)

# Model A (CNN+BiLSTM+Attention)
BATCH_A, LR_A = 64, 2e-3
P1_EPOCHS_A, P2_EPOCHS_A = 30, 20
# Model B (Wav2Vec2-base fine-tune)
W2V_NAME    = "facebook/wav2vec2-base"
N_FREEZE    = 0                         # freeze CNN feature encoder + first N of 12 transformer layers (0 -> all 12 transformer layers fine-tuned)
BATCH_B, LR_B_BACKBONE, LR_B_HEAD = 16, 5e-5, 1e-3
P1_EPOCHS_B, P2_EPOCHS_B = 20, 6
RUN_P2_W2V  = True                      # Protocol 2 for Wav2Vec2 = 5 fine-tunes; the slowest part
N_FOLDS     = 5
NUM_WORKERS = min(4, os.cpu_count() or 2)   # DataLoader workers (augmentation runs on CPU)
RUN_P1_CNN, RUN_P2_CNN = True, True     # set False to skip the (cheap) CNN-BiLSTM runs

if QUICK_RUN:
    P1_EPOCHS_A = P2_EPOCHS_A = P1_EPOCHS_B = P2_EPOCHS_B = 1

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.isdir("/kaggle/input")
if USE_DRIVE and IN_COLAB:
    from google.colab import drive; drive.mount("/content/drive")
    WORK_DIR = "/content/drive/MyDrive/ser_work"
elif IN_KAGGLE:
    WORK_DIR = "/kaggle/working/ser_work"           # persisted as notebook output (20 GB limit)
else:
    WORK_DIR = "/content/ser_work" if IN_COLAB else "./ser_work"
os.makedirs(WORK_DIR, exist_ok=True)
TAG = "quick" if QUICK_RUN else "full"
RESULTS_PATH = os.path.join(WORK_DIR, f"results_{TAG}_v2.json")   # v2 = all-layer fine-tune config (old results are not reused)

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| torch", torch.__version__, "| work dir:", WORK_DIR)

# Persistent results store (resumable): {"p1": {...}, "p2": {...}}
RESULTS = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {"p1": {}, "p2": {}}
def save_results():
    json.dump(RESULTS, open(RESULTS_PATH, "w"))

# %% [markdown]
# ## 1. Datasets & label harmonisation
# Dataset folders are located automatically (Kaggle `/kaggle/input/...`, or `./<name>` on Colab where they are pulled with `opendatasets`). They are the same four corpora as in the project brief.

# %%
def find_root(slug, sub=""):
    """Locate a dataset folder on Kaggle (/kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>) or in ./<slug> (Colab)."""
    cands = [f"/kaggle/input/{slug}", f"./{slug}"] + glob.glob(f"/kaggle/input/datasets/*/{slug}") + glob.glob(f"/kaggle/input/*/{slug}")
    for c in cands:
        if os.path.isdir(c):
            return os.path.join(c, sub) if sub and os.path.isdir(os.path.join(c, sub)) else c
    return None

def locate_all():
    return dict(
        Ravdess=find_root("ravdess-emotional-speech-audio"),
        Crema=find_root("cremad"),
        Savee=find_root("surrey-audiovisual-expressed-emotion-savee"),
        Tess=find_root("toronto-emotional-speech-set-tess"))

roots = locate_all()
if not IN_KAGGLE and not all(roots.values()):          # Colab / local: download from Kaggle with opendatasets
    import opendatasets as od   # prompts for Kaggle username + API key
    for url in ["https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio",
                "https://www.kaggle.com/datasets/ejlok1/cremad",
                "https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee",
                "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"]:
        od.download(url)
    roots = locate_all()
for k, v in roots.items():
    print(f"{k:8s} {v}")
assert all(roots.values()), f"Missing dataset(s): {[k for k, v in roots.items() if not v]} — on Kaggle use 'Add Input' to attach them"
# RAVDESS ships speech *and* song audio; the speech actors folder is what we want (song files would add a 'song' channel)
_rav_speech = glob.glob(os.path.join(roots["Ravdess"], "**", "audio_speech_actors_01-24"), recursive=True)
Ravdess = _rav_speech[0] if _rav_speech else roots["Ravdess"]
Crema, Savee, Tess = roots["Crema"], roots["Savee"], roots["Tess"]

# %% [markdown]
# ### Filename conventions → (emotion, speaker)
# | Dataset | Example | Emotion | Speaker |
# |---|---|---|---|
# | RAVDESS | `03-01-06-01-02-01-12.wav` | 3rd field (1 neutral, **2 calm**, 3 happy, 4 sad, 5 angry, 6 fear, 7 disgust, 8 surprise) | 7th field = actor (24) |
# | CREMA-D | `1001_DFA_ANG_XX.wav` | 3rd token (ANG/DIS/FEA/HAP/NEU/SAD) | 1st token (91 actors) |
# | TESS | `OAF_back_angry.wav` | last token (`ps` = pleasant surprise) | prefix `OAF`/`YAF` (2 speakers) |
# | SAVEE | `DC_a01.wav` | letters after `_` (a, d, f, h, n, sa, su) | prefix `DC/JE/JK/KL` (4 speakers) |
#
# > **Label harmonisation decision:** RAVDESS has an extra **"calm"** class. It is **merged into `neutral`** so that all four corpora share one 7-class label set: `angry, disgust, fear, happy, neutral, sad, surprise`.
# > CREMA-D has no *surprise* clips and RAVDESS/SAVEE/TESS are small, so *surprise* is the rarest class (handled with class weights below).

# %%
def _wavs(folder):
    """All .wav files below `folder` (recursive, case-insensitive extension)."""
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(".wav"))

RAV_MAP = {1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"}  # 2=calm -> neutral
CREMA_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
SAVEE_MAP = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}
TESS_MAP  = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
             "sad": "sad", "ps": "surprise", "surprise": "surprise"}

rows = []
for p in _wavs(Ravdess):
    f = os.path.basename(p)[:-4].split("-")
    if len(f) == 7 and int(f[2]) in RAV_MAP:
        rows.append((p, RAV_MAP[int(f[2])], f"RAVDESS_{f[6]}", "RAVDESS", int(f[2]) == 2))
for p in _wavs(Crema):
    f = os.path.basename(p)[:-4].split("_")
    if len(f) >= 3 and f[2] in CREMA_MAP:
        rows.append((p, CREMA_MAP[f[2]], f"CREMAD_{f[0]}", "CREMA-D", False))
for p in _wavs(Tess):
    f = os.path.basename(p)[:-4].split("_")
    if len(f) >= 3 and f[-1].lower() in TESS_MAP:
        rows.append((p, TESS_MAP[f[-1].lower()], f"TESS_{f[0].upper()}", "TESS", False))
for p in _wavs(Savee):
    f = os.path.basename(p)[:-4].split("_")
    code = re.match(r"[a-z]+", f[1]).group() if len(f) > 1 and re.match(r"[a-z]+", f[1]) else ""
    if code in SAVEE_MAP:
        rows.append((p, SAVEE_MAP[code], f"SAVEE_{f[0].upper()}", "SAVEE", False))

df = pd.DataFrame(rows, columns=["path", "emotion", "speaker", "dataset", "was_calm"])
df["label"] = df.emotion.map(LABEL2ID)
print(f"{len(df):,} clips | {df.speaker.nunique()} speakers | {df.dataset.nunique()} datasets")
print("RAVDESS 'calm' clips merged into neutral:", int(df.was_calm.sum()))
assert df.dataset.nunique() == 4, "All 4 datasets must be present"

if QUICK_RUN:   # small stratified subset for the smoke test
    df = df.sample(frac=1, random_state=SEED).groupby("emotion").head(90)   # <=90 clips per emotion
df = df.reset_index(drop=True)

# %% [markdown]
# ### Class distribution & imbalance handling
# Imbalance is handled with **inverse-frequency class weights** inside the loss (computed from each *training* fold only).

# %%
print(pd.crosstab(df.dataset, df.emotion, margins=True))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.countplot(data=df, x="emotion", order=EMOTIONS, hue="dataset", ax=ax[0]); ax[0].set_title("Class distribution (by dataset)")
sns.countplot(data=df, y="dataset", ax=ax[1]); ax[1].set_title("Clips per dataset")
plt.tight_layout(); plt.show()
cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=df.label.values)
print("Balanced class weights:", dict(zip(EMOTIONS, cw.round(2))))

# %% [markdown]
# ## 2. Pre-processing
# For every clip: **load → resample to 16 kHz mono → `librosa.effects.trim` (silence) → centre-crop / zero-pad to exactly 3 s**.
# The result is cached as `int16` (≈1.2 GB for all clips) so later epochs are fast.
#
# Two feature paths are compared:
# * **(a) log-mel spectrogram** → for the CNN+BiLSTM+Attention model. The mel front-end is implemented *inside* the model as a fixed (non-trainable) conv-STFT, so the exported ONNX model takes a raw waveform.
# * **(b) Wav2Vec2-base embeddings** → the raw waveform goes to `facebook/wav2vec2-base`, whose top layers are fine-tuned.
#
# Both models therefore share the same input `(batch, 48000)` float waveform — which makes serving trivial.

# %%
def to_fixed(y, n=N_SAMPLES):
    """Trim leading/trailing silence, then centre-crop or end-pad to `n` samples.
    The *same* function is used at training and inference time (train/serve parity)."""
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(yt) < 0.1 * SR:          # trimming removed (almost) everything -> keep the original
        yt = y
    if len(yt) >= n:                # centre crop
        s = (len(yt) - n) // 2
        yt = yt[s:s + n]
    else:                           # zero-pad at the end
        yt = np.pad(yt, (0, n - len(yt)))
    return yt.astype(np.float32)

def load_clip(path):
    y, _ = librosa.load(path, sr=SR, mono=True)          # resample to 16 kHz mono
    y = to_fixed(y)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)  # int16 cache

WAV_PATH = os.path.join(WORK_DIR, f"waves_{TAG}.npy")
META_PATH = os.path.join(WORK_DIR, f"meta_{TAG}.csv")
if os.path.exists(WAV_PATH) and os.path.exists(META_PATH) and len(pd.read_csv(META_PATH)) == len(df):
    print("loading cached waveforms")
else:
    t0 = time.time()
    waves = Parallel(n_jobs=-1, batch_size=32)(delayed(load_clip)(p) for p in df.path)
    np.save(WAV_PATH, np.stack(waves)); df.to_csv(META_PATH, index=False)
    print(f"pre-processed {len(df):,} clips in {time.time()-t0:.0f}s")
WAVES = np.load(WAV_PATH, mmap_mode="r")      # (N, 48000) int16, memory-mapped
LABELS = df.label.values
SPEAKERS = df.speaker.values
print(WAVES.shape, WAVES.dtype)

# %%
# Visual sanity check: waveform + log-mel of one clip per emotion
fig, axes = plt.subplots(2, 7, figsize=(22, 5))
for j, e in enumerate(EMOTIONS):
    i = int(np.where(LABELS == LABEL2ID[e])[0][0]); y = WAVES[i].astype(np.float32) / 32768
    axes[0, j].plot(y, lw=.4); axes[0, j].set_title(e); axes[0, j].set_xticks([])
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=64), ref=np.max)
    axes[1, j].imshow(S, origin="lower", aspect="auto"); axes[1, j].set_xticks([])
axes[0, 0].set_ylabel("waveform"); axes[1, 0].set_ylabel("log-mel")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Data augmentation (training data only)
# Implemented **inside the training `Dataset`**, i.e. on the fly *after* the train/val/test split — so no augmented copy of a training clip can ever leak into validation or test (the original notebook augmented before splitting).
#
# | Augmentation | Range | Prob. |
# |---|---|---|
# | Random gain | ×0.5–1.5 | 0.5 |
# | Additive Gaussian (white) noise | SNR 15–35 dB | 0.5 |
# | Pitch shift | ±2 semitones | 0.3 |
# | Time stretch | 0.9×–1.1× | 0.3 |
#
# *Note:* each model normalises its input waveform (zero mean / unit variance) internally so that microphone gain does not matter at serving time; gain augmentation is therefore mostly a no-op for the network but is kept as requested (and it matters when the noise is added at fixed absolute level).

# %%
def _fix_len(y, n=N_SAMPLES):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))

def augment(y, rng):
    """y: float32 waveform, 48000 samples in [-1,1]. Returns an augmented copy of the same length."""
    if rng.random() < 0.5:                                   # random gain
        y = y * rng.uniform(0.5, 1.5)
    if rng.random() < 0.5:                                   # Gaussian noise at random SNR
        snr_db = rng.uniform(15, 35)
        p_sig = np.mean(y ** 2) + 1e-10
        y = y + rng.normal(0, 1, len(y)) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    if rng.random() < 0.3:                                   # pitch shift +-2 semitones
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=rng.uniform(-2, 2))
    if rng.random() < 0.3:                                   # time stretch 0.9-1.1x
        y = librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))
    return _fix_len(np.clip(y, -1, 1)).astype(np.float32)

# Demo on one clip
_rng = np.random.default_rng(0)
_y = WAVES[0].astype(np.float32) / 32768
fig, ax = plt.subplots(1, 3, figsize=(15, 2.5))
for a, (n, v) in zip(ax, [("original", _y), ("augmented #1", augment(_y, _rng)), ("augmented #2", augment(_y, _rng))]):
    a.plot(v, lw=.4); a.set_title(n)
plt.tight_layout(); plt.show()

# %%
class WaveDataset(torch.utils.data.Dataset):
    """Returns (waveform[48000] float32, label). Augmentation only when augment=True (train split)."""
    def __init__(self, idx, augment_on=False):
        self.idx, self.augment_on = np.asarray(idx), augment_on
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        j = self.idx[k]
        y = WAVES[j].astype(np.float32) / 32768.0
        if self.augment_on:
            y = augment(y, np.random.default_rng())          # fresh entropy per call (safe with workers)
        return torch.from_numpy(y), int(LABELS[j])

def make_loader(idx, batch_size, train):
    return torch.utils.data.DataLoader(WaveDataset(idx, augment_on=train), batch_size=batch_size, shuffle=train,
                                       num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                                       drop_last=False, persistent_workers=NUM_WORKERS > 0)

# %% [markdown]
# ## 4. Model architectures
# Both models take a **raw 3 s, 16 kHz waveform** `(B, 48000)` and return **logits** `(B, 7)`.
#
# **A) CNN + BiLSTM + Attention (log-mel).** Fixed conv-STFT → 64-band log-mel → 4 conv blocks (frequency pooled to 4 bands) → BiLSTM over time → additive attention pooling → classifier. SpecAugment (time/frequency masking) is active in training only.
#
# **B) Wav2Vec2-base fine-tune.** CNN feature encoder + feature projection + first `N_FREEZE` transformer layers are **frozen**; the last `12 − N_FREEZE` layers and a small classification head (mean-pool → 256 → 7) are fine-tuned with a lower learning rate for the backbone.

# %%
def utt_norm(x):
    """Per-utterance zero-mean / unit-variance normalisation (also removes mic-gain differences)."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-7)

class MelFrontEnd(nn.Module):
    """Log-mel via fixed conv1d-STFT (cos/sin kernels * Hann window). Pure conv/matmul ops => exports cleanly to ONNX."""
    def __init__(self, sr=SR, n_fft=1024, hop=256, n_mels=64, f_min=20., f_max=8000.):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        win = torch.hann_window(n_fft)
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float(); n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * math.pi * k * n / n_fft
        self.register_buffer("cos_k", (torch.cos(ang) * win).unsqueeze(1))      # (F,1,n_fft)
        self.register_buffer("sin_k", (-torch.sin(ang) * win).unsqueeze(1))
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sr, norm="slaney", mel_scale="slaney")
        self.register_buffer("mel_fb", fb)                                      # (F, n_mels)
    def forward(self, x):                                                       # x: (B,T)
        x = F.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re_, im_ = F.conv1d(x, self.cos_k, stride=self.hop), F.conv1d(x, self.sin_k, stride=self.hop)
        power = re_ ** 2 + im_ ** 2                                             # (B,F,frames)
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)  # (B,n_mels,frames)
        lm = torch.log(mel + 1e-6)
        lm = (lm - lm.mean(dim=(1, 2), keepdim=True)) / torch.sqrt(lm.var(dim=(1, 2), keepdim=True, unbiased=False) + 1e-5)
        return lm.unsqueeze(1)                                                  # (B,1,n_mels,frames)

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_mels=64, hidden=128, dropout=0.3):
        super().__init__()
        self.frontend = MelFrontEnd(n_mels=n_mels)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20, iid_masks=True)
        def block(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(pool), nn.Dropout2d(0.1))
        self.cnn = nn.Sequential(block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 128, (2, 1)), block(128, 128, (2, 1)))
        self.lstm = nn.LSTM(128 * (n_mels // 16), hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1))   # additive attention
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
    def forward(self, wav):
        x = self.frontend(utt_norm(wav))                         # (B,1,M,T)
        if self.training:                                        # SpecAugment (train only)
            x = self.time_mask(self.freq_mask(x))
        x = self.cnn(x)                                          # (B,C,M/16,T/4)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)           # (B,T',C*F)
        h, _ = self.lstm(x)                                      # (B,T',2H)
        w = torch.softmax(self.att(h), dim=1)                    # attention over time
        return self.head((w * h).sum(dim=1))
    def param_groups(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": LR_A}]

from transformers import Wav2Vec2Model
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES, name=W2V_NAME, n_freeze=N_FREEZE, attn="sdpa"):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(name, attn_implementation=attn, layerdrop=0.0,
                                                 mask_time_prob=0.05, mask_time_length=5)
        # --- freezing: conv feature encoder, feature projection, positional conv, first n_freeze transformer layers
        self.w2v.freeze_feature_encoder()
        if n_freeze > 0:    # n_freeze == 0 -> everything except the CNN feature encoder is trained
            for m in [self.w2v.feature_projection, self.w2v.encoder.pos_conv_embed] + list(self.w2v.encoder.layers[:n_freeze]):
                for p in m.parameters(): p.requires_grad = False
        d = self.w2v.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))
    def forward(self, wav):
        h = self.w2v(utt_norm(wav)).last_hidden_state            # (B,T',768)
        return self.head(h.mean(dim=1))                          # mean-pool over time
    def param_groups(self):
        back = [p for p in self.w2v.parameters() if p.requires_grad]
        return [{"params": back, "lr": LR_B_BACKBONE}, {"params": list(self.head.parameters()), "lr": LR_B_HEAD}]

ARCH = {
    "CNN-BiLSTM-Attn": dict(build=lambda: CNNBiLSTMAttention(), amp=False, bs=BATCH_A, ep1=P1_EPOCHS_A, ep2=P2_EPOCHS_A),
    "Wav2Vec2-base":   dict(build=lambda: Wav2Vec2Classifier(), amp=True,  bs=BATCH_B, ep1=P1_EPOCHS_B, ep2=P2_EPOCHS_B),
}
for n, a in ARCH.items():
    m = a["build"]()
    tot = sum(p.numel() for p in m.parameters()); trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    out = m.eval()(torch.randn(2, N_SAMPLES))
    print(f"{n:16s} params {tot/1e6:6.1f}M  trainable {trn/1e6:6.1f}M  output {tuple(out.shape)}")
    del m

# %% [markdown]
# ## 4b. Training / evaluation utilities
# One generic loop for both models: AdamW + One-Cycle LR, class-weighted cross-entropy (label smoothing 0.05), gradient clipping, mixed precision for Wav2Vec2, and **best-validation-epoch checkpointing** (the test set is never used for model selection).

# %%
@torch.no_grad()
def predict_probs(model, idx, bs=64, amp=False):
    model.eval().to(DEVICE)
    out = []
    for xb, _ in make_loader(idx, bs, train=False):
        with torch.autocast(DEVICE.type, enabled=amp and DEVICE.type == "cuda"):
            out.append(torch.softmax(model(xb.to(DEVICE)).float(), dim=-1).cpu())
    return torch.cat(out).numpy()

def train_model(arch, train_idx, val_idx, epochs, tag=""):
    cfg = ARCH[arch]; seed_everything()
    model = cfg["build"]().to(DEVICE)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=LABELS[train_idx])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
    groups = model.param_groups()
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    loader = make_loader(train_idx, cfg["bs"], train=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], epochs=epochs,
                                                steps_per_epoch=len(loader), pct_start=0.15)
    use_amp = cfg["amp"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_acc, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = n = correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=use_amp):
                logits = model(xb); loss = crit(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(yb); n += len(yb); correct += (logits.argmax(1) == yb).sum().item()
        va = accuracy_score(LABELS[val_idx], predict_probs(model, val_idx, amp=use_amp).argmax(1))
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, train_acc=correct / n, val_acc=va))
        print(f"[{arch}{tag}] ep {ep+1:2d}/{epochs} loss {tl/n:.3f} train_acc {correct/n:.3f} val_acc {va:.3f} ({time.time()-t0:.0f}s)")
        if va > best_acc:
            best_acc = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

def plot_history(h, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(h.epoch, h.train_loss); ax[0].set_title(title + " – train loss")
    ax[1].plot(h.epoch, h.train_acc, label="train"); ax[1].plot(h.epoch, h.val_acc, label="val"); ax[1].legend(); ax[1].set_title("accuracy")
    plt.tight_layout(); plt.show()

def plot_cm(y, p, title):
    cm = confusion_matrix(y, p, labels=range(N_CLASSES), normalize="true")
    plt.figure(figsize=(5.5, 4.5)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(title); plt.ylabel("true"); plt.xlabel("predicted"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Evaluation
# ### Protocol 1 — **“Reported Accuracy (Stratified Random Split)”**
# 80 / 10 / 10 train / val / test, **stratified by emotion**, shuffled across *all* speakers.
# This is the headline number for the rubric. **Caveat (printed below):** the same speakers appear in train and test, so the model can exploit speaker/recording-session identity — the number measures *same-speaker* performance.

# %%
all_idx = np.arange(len(df))
tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=LABELS, random_state=SEED, shuffle=True)
va_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=LABELS[tmp_idx], random_state=SEED, shuffle=True)
print(f"train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")
shared = set(SPEAKERS[te_idx]) & set(SPEAKERS[tr_idx])
print(f"Protocol-1 leakage check: {len(shared)}/{len(set(SPEAKERS[te_idx]))} test speakers also appear in train  (expected – this is NOT speaker-independent)")

P1_PRED = {}
def run_protocol1(arch):
    if arch in RESULTS["p1"]:
        print(f"[{arch}] Protocol 1 already done: acc={RESULTS['p1'][arch]['acc']:.4f}"); return
    model, hist = train_model(arch, tr_idx, va_idx, ARCH[arch]["ep1"], tag=" P1")
    probs = predict_probs(model, te_idx, amp=ARCH[arch]["amp"]); pred = probs.argmax(1)
    vprobs = predict_probs(model, va_idx, amp=ARCH[arch]["amp"])      # kept for choosing the ensemble weight on VAL (never on test)
    acc = accuracy_score(LABELS[te_idx], pred); f1 = f1_score(LABELS[te_idx], pred, average="macro")
    RESULTS["p1"][arch] = dict(acc=acc, macro_f1=f1, n_test=len(te_idx), pred=pred.tolist(), true=LABELS[te_idx].tolist(),
                               test_probs=probs.round(5).tolist(), val_probs=vprobs.round(5).tolist(), val_acc=float(accuracy_score(LABELS[va_idx], vprobs.argmax(1))))
    save_results(); plot_history(hist, arch + " P1")
    print(f"\n=== {arch} | Protocol 1 (stratified random split) test accuracy: {acc*100:.2f}% | macro-F1 {f1*100:.2f}% ===")
    del model; torch.cuda.empty_cache()

if RUN_P1_CNN: run_protocol1("CNN-BiLSTM-Attn")

# %%
run_protocol1("Wav2Vec2-base")

# %%
for arch, r in RESULTS["p1"].items():
    print(f"\n##### {arch} — Protocol 1 (Stratified Random Split) #####")
    print(classification_report(r["true"], r["pred"], labels=range(N_CLASSES), target_names=EMOTIONS, digits=3, zero_division=0))
    plot_cm(r["true"], r["pred"], f"{arch} — P1 test confusion matrix")

# %% [markdown]
# ### Protocol 1 add-on — probability-averaging ensemble (CNN-BiLSTM-Attn + Wav2Vec2)
# The ensemble weight is chosen **on the validation set only**; the test set is evaluated once with that weight.

# %%
ens_ok = all(a in RESULTS["p1"] and "val_probs" in RESULTS["p1"][a] for a in ARCH)
if ens_ok:
    vA, vB = [np.array(RESULTS["p1"][a]["val_probs"]) for a in ["CNN-BiLSTM-Attn", "Wav2Vec2-base"]]
    tA, tB = [np.array(RESULTS["p1"][a]["test_probs"]) for a in ["CNN-BiLSTM-Attn", "Wav2Vec2-base"]]
    ws = np.linspace(0, 1, 21)
    val_scores = [accuracy_score(LABELS[va_idx], (w * vA + (1 - w) * vB).argmax(1)) for w in ws]
    w_best = float(ws[int(np.argmax(val_scores))])
    pred = (w_best * tA + (1 - w_best) * tB).argmax(1)
    acc = accuracy_score(LABELS[te_idx], pred)
    RESULTS["p1"]["Ensemble (CNN+W2V)"] = dict(acc=acc, macro_f1=f1_score(LABELS[te_idx], pred, average="macro"), weight_cnn=w_best,
                                              n_test=len(te_idx), pred=pred.tolist(), true=LABELS[te_idx].tolist())
    save_results()
    print(f"Ensemble weight on CNN (picked on val): {w_best:.2f} | Protocol-1 test accuracy: {acc*100:.2f}%")
else:
    print("Ensemble needs both P1 models run with this notebook version (v2).")

# %% [markdown]
# ### Protocol 2 — **“Real-World Generalization Accuracy (Speaker-Independent GroupKFold)”**
# `GroupKFold(5)` with `groups = speaker ID` (121 speakers over the 4 corpora): in every fold **no speaker appears in both train and test** (asserted in code). Early-stopping/checkpoint selection uses a validation set of *held-out speakers carved out of the training fold*, so the test fold is touched only once. Mean ± std over folds is reported, plus the pooled out-of-fold confusion matrix.
#
# For each architecture the best fold model is saved to disk; the winner (higher mean Protocol-2 accuracy) is exported for the real-time demo/API.

# %%
def run_protocol2(arch):
    res = RESULTS["p2"].setdefault(arch, {"folds": []})
    done = {f["fold"] for f in res["folds"]}
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (trn, tst) in enumerate(gkf.split(all_idx, LABELS, groups=SPEAKERS)):
        if k in done:
            print(f"[{arch}] fold {k} already done (acc={[f['acc'] for f in res['folds'] if f['fold']==k][0]:.4f})"); continue
        assert not (set(SPEAKERS[trn]) & set(SPEAKERS[tst])), "speaker leakage between train and test!"
        # inner validation split: ~10 % of the *speakers* of the training fold
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        a, b = next(inner.split(trn, groups=SPEAKERS[trn]))
        model, hist = train_model(arch, trn[a], trn[b], ARCH[arch]["ep2"], tag=f" P2 fold{k}")
        pred = predict_probs(model, tst, amp=ARCH[arch]["amp"]).argmax(1)
        acc = accuracy_score(LABELS[tst], pred); f1 = f1_score(LABELS[tst], pred, average="macro")
        res["folds"].append(dict(fold=k, acc=acc, macro_f1=f1, n_test=len(tst), n_test_speakers=len(set(SPEAKERS[tst])),
                                 pred=pred.tolist(), true=LABELS[tst].tolist()))
        # keep the best fold model for deployment
        best_path = os.path.join(WORK_DIR, f"best_p2_{arch}_{TAG}.pt")
        if acc >= max([f["acc"] for f in res["folds"]]):
            torch.save(model.state_dict(), best_path); res["best_fold"] = k
        save_results()
        print(f"--- {arch} | Protocol 2 fold {k}: unseen-speaker accuracy {acc*100:.2f}% ({len(set(SPEAKERS[tst]))} test speakers) ---")
        del model; torch.cuda.empty_cache()
    accs = [f["acc"] for f in res["folds"]]
    res["mean"], res["std"] = float(np.mean(accs)), float(np.std(accs)); save_results()
    print(f"\n=== {arch} | Protocol 2 (speaker-independent {N_FOLDS}-fold): {res['mean']*100:.2f}% ± {res['std']*100:.2f}% ===")

if RUN_P2_CNN: run_protocol2("CNN-BiLSTM-Attn")

# %%
if RUN_P2_W2V:
    run_protocol2("Wav2Vec2-base")
else:
    print("RUN_P2_W2V=False -> skipping Wav2Vec2 Protocol 2")

# %%
for arch, r in RESULTS["p2"].items():
    if not r.get("folds"): continue
    true = sum([f["true"] for f in r["folds"]], []); pred = sum([f["pred"] for f in r["folds"]], [])
    print(f"\n##### {arch} — Protocol 2 pooled out-of-fold (unseen speakers) #####")
    print(pd.DataFrame([{k: v for k, v in f.items() if k not in ("pred", "true")} for f in r["folds"]]).round(4).to_string(index=False))
    print(classification_report(true, pred, labels=range(N_CLASSES), target_names=EMOTIONS, digits=3, zero_division=0))
    plot_cm(true, pred, f"{arch} — P2 out-of-fold confusion matrix")

# %% [markdown]
# ### Why is Protocol 2 lower than Protocol 1? (and why that is a *feature* of rigorous evaluation)
#
# * **Speaker/session leakage in Protocol 1.** These corpora contain many near-identical recordings per speaker (same sentences, same studio, same microphone, same session). With a random split, the test set contains *the same speakers* seen in training, so a model can score well by recognising speaker voice, recording channel and sentence-specific prosody rather than purely the emotion.
# * **Protocol 2 removes that shortcut.** `GroupKFold` by speaker guarantees every test speaker is **completely unseen**, which is exactly the situation of a deployed app: a new person speaks into a new microphone.
# * **Acted, small-speaker corpora.** TESS (2 speakers) and SAVEE (4 speakers) provide very little speaker diversity; CREMA-D is crowd-rated and contains ambiguous emotional expressions (its human-rater accuracy is only ≈ 40–60 %), and RAVDESS/SAVEE/TESS are acted at higher intensity than natural speech. Cross-speaker, cross-corpus generalisation is therefore genuinely harder.
# * **Interpretation.** The *random-split* figure is the rubric/“reported” number and represents performance *when speakers are shared*; the *speaker-independent* figure is the **honest estimate of real-world / real-time performance**, so the real-time demo and API use the Protocol 2 model.
#
# A gap between the two is expected and healthy; a model that scores the same on both would be suspicious (or the data too easy).

# %% [markdown]
# ### Comparison table & chart (both protocols, both models)

# %%
rows = []
for arch in ARCH:
    p1 = RESULTS["p1"].get(arch); p2 = RESULTS["p2"].get(arch, {})
    rows.append({"Model": arch,
                 "Protocol 1 – Stratified random split (test acc %)": round(p1["acc"] * 100, 2) if p1 else np.nan,
                 "Protocol 2 – Speaker-independent GroupKFold (mean acc %)": round(p2["mean"] * 100, 2) if "mean" in p2 else np.nan,
                 "Protocol 2 std (%)": round(p2["std"] * 100, 2) if "std" in p2 else np.nan})
cmp_df = pd.DataFrame(rows); cmp_df["Gap (P1 − P2, pts)"] = (cmp_df.iloc[:, 1] - cmp_df.iloc[:, 2]).round(2)
display(cmp_df.set_index("Model"))

fig, ax = plt.subplots(figsize=(8, 4.5)); w = 0.35; x = np.arange(len(cmp_df))
b1 = ax.bar(x - w/2, cmp_df.iloc[:, 1], w, label="Protocol 1: stratified random split (same speakers in train/test)", color="#4C78A8")
b2 = ax.bar(x + w/2, cmp_df.iloc[:, 2], w, yerr=cmp_df["Protocol 2 std (%)"].fillna(0), capsize=4,
            label="Protocol 2: speaker-independent GroupKFold (unseen speakers)", color="#F58518")
for b in list(b1) + list(b2):
    if not np.isnan(b.get_height()): ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cmp_df.Model); ax.set_ylim(0, 108); ax.set_ylabel("accuracy (%)")
ax.axhline(95, ls="--", c="gray", lw=.8); ax.text(len(x) - .5, 95.8, "95 % target", ha="right", fontsize=8, color="gray")
ax.set_title("Two evaluation protocols – never to be conflated"); ax.legend(loc="lower center", fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Real-time inference demo
# **Model choice:** the architecture with the higher **Protocol 2** mean accuracy; its best speaker-independent fold model is exported.
# Export formats: **ONNX** (served by ONNX Runtime in the API) + **TorchScript**. A dynamic-INT8 ONNX is produced too (smaller / faster on CPU) and checked for prediction agreement.

# %%
p2_done = {a: r["mean"] for a, r in RESULTS["p2"].items() if "mean" in r}
assert p2_done, "Run Protocol 2 first"
WINNER = max(p2_done, key=p2_done.get)
print("Protocol-2 means:", {a: f"{v*100:.2f}%" for a, v in p2_done.items()}, "->  deploying:", WINNER)

# Rebuild the winner (eager attention is the most ONNX-friendly for Wav2Vec2) and load the best speaker-independent fold
if WINNER == "Wav2Vec2-base":
    export_model = Wav2Vec2Classifier(attn="eager")
else:
    export_model = CNNBiLSTMAttention()
export_model.load_state_dict(torch.load(os.path.join(WORK_DIR, f"best_p2_{WINNER}_{TAG}.pt"), map_location="cpu"))
export_model.eval().cpu()

EXPORT_DIR = os.path.join(WORK_DIR, "export"); os.makedirs(EXPORT_DIR, exist_ok=True)
ONNX_PATH, INT8_PATH, TS_PATH = [os.path.join(EXPORT_DIR, n) for n in ("model.onnx", "model_int8.onnx", "model_ts.pt")]
dummy = torch.randn(1, N_SAMPLES)

# ---- ONNX (waveform -> logits). Legacy tracer (dynamo=False) is the most reliable for Wav2Vec2.
export_kwargs = dict(input_names=["waveform"], output_names=["logits"], dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(export_model, (dummy,), ONNX_PATH, dynamo=False, **export_kwargs)
except TypeError:
    torch.onnx.export(export_model, (dummy,), ONNX_PATH, **export_kwargs)

# ---- TorchScript
try:
    torch.jit.trace(export_model, dummy, check_trace=False).save(TS_PATH); print("TorchScript saved")
except Exception as e:
    print("TorchScript trace skipped:", repr(e)[:200])

# ---- INT8 dynamic quantisation (optional)
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
try:
    quantize_dynamic(ONNX_PATH, INT8_PATH, weight_type=QuantType.QInt8)
except Exception as e:
    print("INT8 quantisation skipped:", repr(e)[:200]); INT8_PATH = None

json.dump({"labels": EMOTIONS, "sample_rate": SR, "window_samples": N_SAMPLES, "top_db": TOP_DB, "architecture": WINNER,
           "protocol2_mean_acc": p2_done[WINNER], "protocol1_acc": RESULTS["p1"].get(WINNER, {}).get("acc")},
          open(os.path.join(EXPORT_DIR, "labels.json"), "w"), indent=2)

# ---- parity check: PyTorch vs ONNX vs INT8 on a handful of test clips + CPU latency
chk = WaveDataset(te_idx[:16]); xb = torch.stack([chk[i][0] for i in range(len(chk))])
with torch.no_grad(): ref = export_model(xb).numpy()
for name, path in [("onnx fp32", ONNX_PATH), ("onnx int8", INT8_PATH)]:
    if path is None: continue
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"waveform": xb.numpy()})[0]
    t0 = time.time(); [sess.run(None, {"waveform": xb[:1].numpy()}) for _ in range(5)]; lat = (time.time() - t0) / 5 * 1000
    print(f"{name}: size {os.path.getsize(path)/1e6:6.1f} MB | max|Δlogit| {np.abs(out-ref).max():.2e} | argmax agreement {(out.argmax(1)==ref.argmax(1)).mean():.0%} | {lat:.0f} ms / 3 s window (CPU)")

# %%
# ---------- voice-activity detection + sliding-window streaming ----------
from silero_vad import load_silero_vad, get_speech_timestamps
VAD = load_silero_vad()
SERVE_PATH = INT8_PATH if INT8_PATH and os.path.exists(INT8_PATH) else ONNX_PATH
SESS = ort.InferenceSession(SERVE_PATH, providers=["CPUExecutionProvider"])

def classify_window(y):
    """y: float32 mono 16 kHz audio (any length ≤ ~3 s) -> (emotion, confidence, probs). Uses the training pre-processing."""
    x = to_fixed(y)[None]
    logits = SESS.run(None, {"waveform": x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return EMOTIONS[int(p.argmax())], float(p.max()), p

def stream_emotions(y, win_s=3.0, hop_s=1.5, min_speech=0.3, realtime=False, verbose=True):
    """Slide a `win_s` window with 50 % overlap (hop = win/2). A window is classified only if silero-VAD finds
    ≥ `min_speech` fraction of speech inside it. Returns a list of dicts."""
    ts = get_speech_timestamps(torch.from_numpy(y), VAD, sampling_rate=SR)
    speech = np.zeros(len(y), dtype=bool)
    for t in ts: speech[t["start"]:t["end"]] = True
    W, H, out = int(win_s * SR), int(hop_s * SR), []
    for s in range(0, max(1, len(y) - W // 2), H):
        seg, sp = y[s:s + W], speech[s:s + W]
        rec = {"t_start": s / SR, "t_end": min(s + W, len(y)) / SR, "speech_frac": float(sp.mean()) if len(sp) else 0.0}
        if rec["speech_frac"] >= min_speech:
            t0 = time.time(); emo, conf, p = classify_window(seg)
            rec.update(emotion=emo, confidence=conf, probs=p, latency_ms=(time.time() - t0) * 1000)
        else:
            rec.update(emotion=None, confidence=None, probs=None, latency_ms=0.0)
        out.append(rec)
        if verbose:
            msg = (f"{rec['emotion']:9s} {rec['confidence']*100:5.1f}%  ({rec['latency_ms']:.0f} ms)" if rec["emotion"] else "— no speech —")
            print(f"[{rec['t_start']:5.1f}s – {rec['t_end']:5.1f}s]  {msg}")
        if realtime: time.sleep(hop_s)
    return out

# Demo 1: concatenate one test clip per emotion with silence gaps to simulate a conversation stream
rng = np.random.default_rng(1); demo, truth = [], []
for e in ["angry", "happy", "sad", "neutral", "fear"]:
    cand = [i for i in te_idx if LABELS[i] == LABEL2ID[e]]
    if not cand: continue
    demo += [WAVES[rng.choice(cand)].astype(np.float32) / 32768, np.zeros(int(1.0 * SR), np.float32)]; truth.append(e)
stream = np.concatenate(demo)
print("ground-truth order:", truth)
res = stream_emotions(stream)

# Confidence timeline plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(np.arange(len(stream)) / SR, stream, lw=.3, c="gray")
for r in res:
    if r["emotion"]:
        ax.axvspan(r["t_start"], r["t_end"], alpha=.12 + .25 * r["confidence"], color=sns.color_palette("tab10")[EMOTIONS.index(r["emotion"])])
        ax.text((r["t_start"] + r["t_end"]) / 2, 0.8 * np.abs(stream).max(), f"{r['emotion']}\n{r['confidence']*100:.0f}%", ha="center", fontsize=8)
ax.set_xlabel("time (s)"); ax.set_title("Sliding-window (3 s, 50 % overlap) emotion stream with VAD gating"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### Live microphone / uploaded audio
# * **Microphone (Colab):** run the cell below, allow the browser mic, speak for N seconds; the recording is streamed through the same VAD + sliding-window pipeline.
# * **Upload:** set `AUDIO_FILE` to any wav/mp3/m4a path (e.g. via `files.upload()`).
#
# The browser captures audio chunks in the page; Colab cannot give true sample-level streaming to Python, so the recording is processed right after capture — latency per window (printed above) is a few tens/hundreds of ms on CPU, i.e. fully real-time capable.

# %%
AUDIO_FILE = None          # e.g. "my_speech.wav"; leave None to use the microphone recorder below

def record_from_browser(seconds=8):
    """Colab-only: record from the browser microphone, return float32 16 kHz mono numpy array."""
    from IPython.display import Javascript, display
    from google.colab import output
    from base64 import b64decode
    import subprocess
    js = Javascript("""
    async function rec(sec){
      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      const mr = new MediaRecorder(stream); const chunks=[];
      mr.ondataavailable = e => chunks.push(e.data);
      const done = new Promise(r => mr.onstop = r);
      mr.start(); await new Promise(r => setTimeout(r, sec*1000)); mr.stop(); await done;
      stream.getTracks().forEach(t=>t.stop());
      const buf = await new Blob(chunks).arrayBuffer();
      let bin=''; const b=new Uint8Array(buf); for (let i=0;i<b.length;i++) bin+=String.fromCharCode(b[i]);
      return btoa(bin);
    }""")
    display(js); print(f"🎙️ recording {seconds}s ... speak now"); b64 = output.eval_js(f"rec({seconds})")
    open("rec.webm", "wb").write(b64decode(b64))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "rec.webm", "-ar", str(SR), "-ac", "1", "rec.wav"], check=True)
    return librosa.load("rec.wav", sr=SR, mono=True)[0]

if AUDIO_FILE:
    live = librosa.load(AUDIO_FILE, sr=SR, mono=True)[0]
elif IN_COLAB:
    live = record_from_browser(8)
else:
    live = stream   # fallback: reuse the demo stream when not on Colab
_ = stream_emotions(live, realtime=False)

# %% [markdown]
# ## 7. Final report

# %%
from IPython.display import Markdown, display
lines = ["### Final results (all numbers measured by this notebook)\n"]
for arch in ARCH:
    p1 = RESULTS["p1"].get(arch); p2 = RESULTS["p2"].get(arch, {})
    if p1 and "mean" in p2:
        lines.append(f"* **{arch}** — Stratified random-split test accuracy: **{p1['acc']*100:.1f}%** | "
                     f"Speaker-independent (unseen-speaker, {N_FOLDS}-fold GroupKFold) accuracy: **{p2['mean']*100:.1f}% ± {p2['std']*100:.1f}%**")
lines.append(f"\n**Deployed model (best speaker-independent mean): {WINNER}** → `{os.path.basename(SERVE_PATH)}`.\n")
lines.append("The random-split number reflects performance when the same speakers appear in training and test; "
             "the speaker-independent number is the honest estimate of real-world / real-time performance.")
target = RESULTS["p1"].get(WINNER, {}).get("acc", 0)
lines.append(f"\n95 % target on the random split: **{'met' if target >= 0.95 else 'NOT met'}** by the deployed architecture "
             f"({target*100:.1f}%). " + ("" if target >= 0.95 else
             "If short of target, options: more epochs, un-freeze more Wav2Vec2 layers, or add EMO-DB / MELD / IEMOCAP (never fewer than 4 datasets)."))
display(Markdown("\n".join(lines)))
print("\nArtifacts to download for deployment:", [os.path.join(EXPORT_DIR, n) for n in ("model_int8.onnx", "model.onnx", "labels.json")])

# %%
# Convenience: collect the deployment artefacts in one place
import shutil
OUT_DIR = "/kaggle/working/deploy_artifacts" if IN_KAGGLE else os.path.join(WORK_DIR, "deploy_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)
for f in [INT8_PATH, ONNX_PATH, os.path.join(EXPORT_DIR, "labels.json")]:
    if f and os.path.exists(f): shutil.copy(f, OUT_DIR)
print("Artefacts ready in", OUT_DIR, "->", os.listdir(OUT_DIR))
if IN_COLAB:
    from google.colab import files
    for f in os.listdir(OUT_DIR): files.download(os.path.join(OUT_DIR, f))
else:
    print("Kaggle: open the notebook's *Output* tab (after Save Version) to download these files.")
