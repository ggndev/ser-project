# %% [markdown]
# # Speech Emotion Recognition with Self-Supervised Speech Transformers
# ### A speaker-independent study on TESS, RAVDESS, SAVEE and CREMA-D — with a deployable real-time API
#
# **Notebook type:** project report (every section = explanation → code → result → discussion)
# **Runtime:** Kaggle Notebook · *Accelerator: GPU T4 ×2 (one GPU is used)* · *Internet: On* (to download pretrained models)
# **Stack:** PyTorch · torchaudio · 🤗 Transformers · librosa · scikit-learn · silero-VAD · ONNX Runtime
#
# ---
# ## Abstract
# Recognising emotion from speech is a core problem in human–computer interaction, call-centre analytics and assistive technology.
# We build a 7-class speech emotion recogniser (*angry, disgust, fear, happy, neutral, sad, surprise*) from **four public acted-speech corpora** (≈12 k utterances, 121 speakers).
# We compare a lightweight **CNN + BiLSTM + attention** network on log-mel spectrograms with three **self-supervised transformer backbones** (Wav2Vec 2.0, HuBERT, WavLM) fine-tuned end-to-end with a learned weighted sum of transformer layers.
# Every model is evaluated under **two clearly separated protocols**: (1) a *stratified random split*, the usual headline number in the literature, and (2) a *speaker-independent 5-fold GroupKFold*, which measures what happens for a person the model has never heard.
# The best speaker-independent model is exported to **ONNX**, quantised, served by a **FastAPI** service on a free cloud tier and consumed by an **Android app**.
# *The measured numbers are produced by the code and collected in Section 12 (“Final results”), which also prints a ready-to-paste abstract with the actual values.*
#
# ## Contents
# | § | Section | § | Section |
# |---|---|---|---|
# | 1 | Introduction & objectives | 8 | Evaluation protocols & metrics |
# | 2 | Environment & configuration | 9 | Experiment 1 – random-split results |
# | 3 | Datasets & label harmonisation | 10 | Experiment 2 – speaker-independent results |
# | 4 | Pre-processing | 11 | Analysis & discussion |
# | 5 | Data augmentation | 12 | **Final results (auto-generated)** |
# | 6 | Model architectures | 13 | Deployment (ONNX · API · Android) |
# | 7 | Training procedure | 14 | Conclusion, limitations, future work, references |
#
# **How to run.** *Add Input* → add the four datasets (`ravdess-emotional-speech-audio`, `cremad`, `surrey-audiovisual-expressed-emotion-savee`, `toronto-emotional-speech-set-tess`) → Internet **On** → GPU **T4 ×2** → *Run All* (or *Save Version ▸ Save & Run All* for an unattended run).
# Everything long-running is **resumable** (results are stored in `results_*.json` after each model / fold). Set `QUICK_RUN = True` in §2 for a ~15-minute smoke test.

# %% [markdown]
# ## 1. Introduction & objectives
#
# **Problem.** Given a short speech clip, predict the speaker's emotional state. Emotion is carried by *prosody* (pitch, energy, rhythm), *voice quality* and, to a lesser degree, by the words themselves. The task is hard because (i) emotions are subjective and overlap, (ii) public datasets are small and *acted*, and (iii) the same emotion sounds different across speakers, languages and microphones.
#
# **Objectives**
# 1. Build an accurate 7-class classifier using **≥ 4 public datasets**.
# 2. Quantify **both** the usual random-split accuracy **and** the honest *unseen-speaker* accuracy, and explain the gap.
# 3. Compare classical-style (CNN/BiLSTM on spectrograms) and modern (self-supervised transformer) approaches.
# 4. Deliver a **real-time, free-tier deployment** (ONNX model → REST API → Android app).
#
# **Key design decisions (and why)**
# | Decision | Reason |
# |---|---|
# | Merge RAVDESS *calm* into *neutral* | the other three corpora have no *calm* class; a common 7-label set is required |
# | Augmentation only on the training split, *after* splitting | augmenting before splitting puts near-duplicates of test clips into training (data leakage) |
# | Report two protocols | random split overestimates real-world accuracy because speakers repeat across train/test |
# | Model selection on validation data only | the test set must be used once, otherwise the reported accuracy is optimistically biased |
# | Fixed 3 s input window | simple batching, constant latency for the real-time system |

# %% [markdown]
# ## 2. Environment & configuration
# All hyper-parameters live in one cell so that experiments are reproducible. The defaults fit a single Kaggle session (≈ 6–8 GPU-hours on a T4).
#
# | Stage | Approx. time on a T4 |
# |---|---|
# | Pre-processing (12 k clips, parallel) | 5–10 min |
# | Protocol 1: CNN-BiLSTM-Attn | 10–15 min |
# | Protocol 1: three transformer backbones (12 epochs each) | ≈ 3 h |
# | Protocol 2: CNN (5 folds) + best transformer (5 folds × 6 epochs) | ≈ 2–2.5 h |
# | Export, quantisation, plots | 10 min |

# %%
# Colab/Kaggle already ship torch / torchaudio / numpy / pandas / scikit-learn / transformers; we add the rest (Internet must be On).
!pip -q install librosa soundfile opendatasets "transformers>=4.40" accelerate onnx onnxscript onnxruntime silero-vad seaborn joblib

# %%
import os, sys, json, math, random, time, glob, re, copy, itertools, shutil, warnings
import numpy as np
import pandas as pd
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import librosa, soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

# ======================= CONFIGURATION (edit here) =======================
QUICK_RUN   = False      # True -> tiny subset & 1 epoch: checks the whole pipeline in ~15 min
SEED        = 42
SR          = 16_000                     # target sample rate (Hz) - what all pretrained speech models expect
DUR         = 3.0                        # fixed analysis window (s)
N_SAMPLES   = int(SR * DUR)              # 48 000 samples
TOP_DB      = 30                         # silence threshold (dB below peak) for librosa.effects.trim
EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID    = {e: i for i, e in enumerate(EMOTIONS)}
N_CLASSES   = len(EMOTIONS)

# --- Model A: CNN + BiLSTM + attention on log-mel
BATCH_A, LR_A = 64, 2e-3
P1_EPOCHS_A, P2_EPOCHS_A = 30, 20

# --- Model B: self-supervised transformer backbones (all fine-tuned end-to-end except the CNN feature encoder)
HF_IDS = {                                # name -> Hugging Face model id
    "Wav2Vec2-base":   "facebook/wav2vec2-base",
    "HuBERT-base":     "facebook/hubert-base-ls960",
    "WavLM-base-plus": "microsoft/wavlm-base-plus",
}
BATCH_B, LR_B_BACKBONE, LR_B_HEAD = 16, 5e-5, 1e-3
P1_EPOCHS_B, P2_EPOCHS_B = 12, 6          # P2 uses fewer epochs because it is repeated for 5 folds

# --- Protocol 2
N_FOLDS     = 5
P2_TRANSFORMERS = 1                       # how many of the best transformers (by P1 *validation* acc) also get Protocol 2
ENABLE_ENSEMBLE = False                   # optional: average transformer probabilities (weights/subset chosen on validation)
NUM_WORKERS = min(4, os.cpu_count() or 2) # CPU workers for on-the-fly augmentation

if QUICK_RUN:
    P1_EPOCHS_A = P2_EPOCHS_A = P1_EPOCHS_B = P2_EPOCHS_B = 1

IN_KAGGLE = os.path.isdir("/kaggle/input")
IN_COLAB  = "google.colab" in sys.modules
WORK_DIR  = "/kaggle/working/ser_report" if IN_KAGGLE else ("/content/ser_report" if IN_COLAB else "./ser_report")
FIG_DIR   = os.path.join(WORK_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TAG = "quick" if QUICK_RUN else "full"
RESULTS_PATH = os.path.join(WORK_DIR, f"results_{TAG}.json")

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Persistent, resumable results store
RESULTS = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {"p1": {}, "p2": {}}
def save_results(): json.dump(RESULTS, open(RESULTS_PATH, "w"))

def show(name):
    """Save the current matplotlib figure as a report-ready PNG (200 dpi) and display it."""
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200, bbox_inches="tight"); plt.show()

# Reproducibility information (copy into the report's 'Experimental setup')
print("Device      :", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
print("Versions    : torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| librosa", librosa.__version__)
import transformers; print("transformers:", transformers.__version__, "| seed", SEED, "| work dir", WORK_DIR)

# %% [markdown]
# ## 3. Datasets & label harmonisation
#
# We use four widely-cited **acted** emotional speech corpora:
#
# | Corpus | Language / speakers | Clips | Emotions | Notes |
# |---|---|---|---|---|
# | **TESS** (Toronto) | English · 2 female | 2 800 | 7 (incl. *pleasant surprise*) | one carrier phrase “Say the word …”, very clean studio audio |
# | **RAVDESS** | English (North-American) · 24 actors | 1 440 speech clips | 8 (incl. *calm*) | two intensity levels, professional actors |
# | **SAVEE** | British English · 4 male | 480 | 7 | very few speakers |
# | **CREMA-D** | English · 91 actors | 7 442 | 6 (*no surprise*) | diverse ethnicity/age; labels from crowd-sourced raters → noisy |
#
# **Label mapping.** All corpora are mapped to a common set `angry, disgust, fear, happy, neutral, sad, surprise`.
# * RAVDESS **“calm” → “neutral”** (the other corpora have no calm class; calm and neutral are acoustically close). *This is a deliberate, documented choice.*
# * TESS “pleasant surprise” (`ps`) → *surprise*.
# * Speaker IDs are parsed from file names; they are needed later for the speaker-independent protocol.
#
# | Dataset | File-name pattern → emotion | → speaker |
# |---|---|---|
# | RAVDESS | `03-01-06-01-02-01-12.wav`: 3rd field (1 neutral, **2 calm**, 3 happy, 4 sad, 5 angry, 6 fear, 7 disgust, 8 surprise) | 7th field (actor 01–24) |
# | CREMA-D | `1001_DFA_ANG_XX.wav`: 3rd token (ANG DIS FEA HAP NEU SAD) | 1st token (1001–1091) |
# | TESS | `OAF_back_angry.wav`: last token | prefix `OAF` / `YAF` |
# | SAVEE | `DC_a01.wav`: letters after `_` (a d f h n sa su) | prefix `DC JE JK KL` |

# %%
def find_root(slug):
    """Locate a dataset on Kaggle (/kaggle/input/<slug> or .../datasets/<owner>/<slug>) or in ./<slug>."""
    cands = [f"/kaggle/input/{slug}", f"./{slug}"] + glob.glob(f"/kaggle/input/datasets/*/{slug}") + glob.glob(f"/kaggle/input/*/{slug}")
    return next((c for c in cands if os.path.isdir(c)), None)

def locate_all():
    return dict(Ravdess=find_root("ravdess-emotional-speech-audio"), Crema=find_root("cremad"),
                Savee=find_root("surrey-audiovisual-expressed-emotion-savee"), Tess=find_root("toronto-emotional-speech-set-tess"))

roots = locate_all()
if not IN_KAGGLE and not all(roots.values()):            # Colab / local: download via the Kaggle API
    import opendatasets as od
    for url in ["https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio",
                "https://www.kaggle.com/datasets/ejlok1/cremad",
                "https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee",
                "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"]:
        od.download(url)
    roots = locate_all()
assert all(roots.values()), f"Missing dataset(s): {[k for k, v in roots.items() if not v]} - use 'Add Input' on Kaggle"
# RAVDESS contains speech AND song audio; we only want the speech actors
_rav = glob.glob(os.path.join(roots["Ravdess"], "**", "audio_speech_actors_01-24"), recursive=True)
Ravdess, Crema, Savee, Tess = (_rav[0] if _rav else roots["Ravdess"]), roots["Crema"], roots["Savee"], roots["Tess"]

def _wavs(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(".wav"))

RAV_MAP   = {1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"}   # 2 = calm -> neutral
CREMA_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
SAVEE_MAP = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}
TESS_MAP  = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
             "sad": "sad", "ps": "surprise", "surprise": "surprise"}

rows = []   # (path, emotion, speaker, dataset, was_calm)
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
    m = re.match(r"[a-z]+", f[1]) if len(f) > 1 else None
    if m and m.group() in SAVEE_MAP:
        rows.append((p, SAVEE_MAP[m.group()], f"SAVEE_{f[0].upper()}", "SAVEE", False))

df = pd.DataFrame(rows, columns=["path", "emotion", "speaker", "dataset", "was_calm"])
df["label"] = df.emotion.map(LABEL2ID)
assert df.dataset.nunique() == 4, "all four datasets must be present"
if QUICK_RUN:   # small stratified subset for the smoke test
    df = df.sample(frac=1, random_state=SEED).groupby("emotion").head(90)
df = df.reset_index(drop=True)
print(f"{len(df):,} clips | {df.speaker.nunique()} speakers | {df.dataset.nunique()} datasets | RAVDESS 'calm' merged into neutral: {int(df.was_calm.sum())}")

# %% [markdown]
# ### 3.1 Exploratory data analysis
# The cross-tabulation and plots below document **class balance** and **dataset composition** — both matter for interpreting the results:
# * CREMA-D dominates the corpus (≈ 60 %), so the *combined* score is largely a CREMA-D score.
# * *Surprise* is the rarest class and does not exist in CREMA-D.
# * TESS and SAVEE have only 2 and 4 speakers → little speaker diversity.

# %%
print(pd.crosstab(df.dataset, df.emotion, margins=True))
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
sns.countplot(data=df, x="emotion", order=EMOTIONS, hue="dataset", ax=ax[0]); ax[0].set_title("(a) Class distribution by dataset")
sns.countplot(data=df, y="dataset", ax=ax[1], order=df.dataset.value_counts().index); ax[1].set_title("(b) Clips per dataset")
spk = df.groupby("dataset").speaker.nunique().sort_values(); spk.plot.barh(ax=ax[2], color="#4C78A8"); ax[2].set_title("(c) Speakers per dataset")
plt.tight_layout(); show("fig01_dataset_overview")
cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=df.label.values)
print("Inverse-frequency class weights:", dict(zip(EMOTIONS, cw.round(2))))

# %% [markdown]
# **Handling imbalance.** We use **inverse-frequency class weights** inside the loss, computed from each *training* fold only:
# $$ w_c = \frac{N}{K\,n_c},\qquad \mathcal{L}=-\sum_{c=1}^{K} w_c\,\tilde y_c \log p_c,\qquad \tilde y_c=(1-\varepsilon)\,y_c+\frac{\varepsilon}{K} $$
# where $N$ is the number of training clips, $K=7$ classes, $n_c$ the clips of class $c$, and $\varepsilon=0.05$ is **label smoothing** (it discourages over-confident predictions, which also makes the reported confidence more meaningful).

# %% [markdown]
# ## 4. Pre-processing
#
# Every clip goes through the **same function at training and at inference time** (this “train/serve parity” is essential for a deployed system):
#
# 1. **Load & resample** to 16 kHz mono (all pretrained speech models expect 16 kHz).
# 2. **Trim silence** with `librosa.effects.trim(top_db=30)`: frames more than 30 dB below the peak at the start/end are removed, so the model sees speech, not room noise.
# 3. **Fix the length to 3 s**: centre-crop longer clips, zero-pad shorter ones. A fixed window simplifies batching and gives constant inference latency.
# 4. **Per-utterance normalisation** (zero mean, unit variance) *inside* the model, which removes microphone-gain differences between datasets and phones.
#
# Two **feature paths** are compared:
# * **(a) log-mel spectrogram → CNN + BiLSTM + attention.** A mel filter-bank imitates the ear's non-linear frequency resolution: $m = 2595\log_{10}(1+f/700)$; we use $\log(\mathbf{M}\,|\mathrm{STFT}(x)|^2+\varepsilon)$ with 64 bands (window 1024, hop 256 ≈ 16 ms).
# * **(b) Self-supervised speech representations** (Wav2Vec 2.0 / HuBERT / WavLM): the network consumes the raw waveform and learned contextual features replace hand-made ones.
#
# Both models therefore have the same interface — a `(batch, 48000)` waveform → 7 logits — which makes ONNX export and serving trivial.

# %%
def to_fixed(y, n=N_SAMPLES):
    """Trim leading/trailing silence, then centre-crop or end-pad to exactly `n` samples (same code is used by the API)."""
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(yt) < 0.1 * SR:                 # trimming removed (almost) everything -> keep the original
        yt = y
    if len(yt) >= n:                       # centre crop
        s = (len(yt) - n) // 2
        yt = yt[s:s + n]
    else:                                  # zero-pad at the end
        yt = np.pad(yt, (0, n - len(yt)))
    return yt.astype(np.float32)

def load_clip(path):
    y, _ = librosa.load(path, sr=SR, mono=True)            # resample to 16 kHz mono
    return (np.clip(to_fixed(y), -1, 1) * 32767).astype(np.int16)    # int16 cache saves 50 % memory

WAV_PATH, META_PATH = os.path.join(WORK_DIR, f"waves_{TAG}.npy"), os.path.join(WORK_DIR, f"meta_{TAG}.csv")
if os.path.exists(WAV_PATH) and os.path.exists(META_PATH) and len(pd.read_csv(META_PATH)) == len(df):
    print("loading cached waveforms")
else:
    t0 = time.time()
    waves = Parallel(n_jobs=-1, batch_size=32)(delayed(load_clip)(p) for p in df.path)
    np.save(WAV_PATH, np.stack(waves)); df.to_csv(META_PATH, index=False)
    print(f"pre-processed {len(df):,} clips in {time.time()-t0:.0f}s")
WAVES = np.load(WAV_PATH, mmap_mode="r")            # (N, 48000) int16, memory-mapped
LABELS, SPEAKERS, DATASETS = df.label.values, df.speaker.values, df.dataset.values
print(WAVES.shape, WAVES.dtype)

# %%
# Figure: one example per emotion – waveform (top) and log-mel (bottom)
fig, axes = plt.subplots(2, 7, figsize=(22, 5))
for j, e in enumerate(EMOTIONS):
    i = int(np.where(LABELS == LABEL2ID[e])[0][0]); y = WAVES[i].astype(np.float32) / 32768
    axes[0, j].plot(y, lw=.4); axes[0, j].set_title(f"{e}\n({df.dataset[i]})"); axes[0, j].set_xticks([])
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=64), ref=np.max)
    axes[1, j].imshow(S, origin="lower", aspect="auto", cmap="magma"); axes[1, j].set_xticks([])
axes[0, 0].set_ylabel("waveform"); axes[1, 0].set_ylabel("log-mel (64 bands)")
plt.tight_layout(); show("fig02_waveforms_and_logmel")

# %% [markdown]
# ## 5. Data augmentation (training split only)
# Augmentation synthesises new training examples by perturbing speech in ways that should *not* change the emotion, which reduces over-fitting on small corpora and improves robustness to microphones and noise.
#
# | Augmentation | Range | Probability | Rationale |
# |---|---|---|---|
# | Random gain | ×0.5 – 1.5 | 0.5 | different recording levels |
# | Additive white Gaussian noise | SNR 15 – 35 dB | 0.5 | noisy rooms / cheap microphones |
# | Pitch shift | ± 2 semitones | 0.3 | different speaker pitch ranges |
# | Time stretch | 0.9 × – 1.1 × | 0.3 | different speaking rates |
#
# **No data leakage:** augmentation is applied *on the fly inside the training `Dataset`*, i.e. **after** the train/validation/test split. Validation and test clips are never augmented. (Augmenting the whole corpus first, as in many tutorials, would place perturbed copies of a test clip in the training set and inflate accuracy.)
# Two additional regularisers are active during training only: **SpecAugment** time/frequency masking in the CNN model (Park et al., 2019) and **time-masking** inside the transformer backbones.

# %%
def _fix_len(y, n=N_SAMPLES):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))

def augment(y, rng):
    """y: float32 waveform (48000 samples, [-1, 1]) -> augmented copy of the same length."""
    if rng.random() < 0.5:                                   # random gain
        y = y * rng.uniform(0.5, 1.5)
    if rng.random() < 0.5:                                   # white Gaussian noise at a random SNR
        snr_db = rng.uniform(15, 35)
        y = y + rng.normal(0, 1, len(y)) * np.sqrt((np.mean(y ** 2) + 1e-10) / 10 ** (snr_db / 10))
    if rng.random() < 0.3:                                   # pitch shift +-2 semitones
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=rng.uniform(-2, 2))
    if rng.random() < 0.3:                                   # time stretch 0.9-1.1x
        y = librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))
    return _fix_len(np.clip(y, -1, 1)).astype(np.float32)

_rng = np.random.default_rng(0); _y = WAVES[0].astype(np.float32) / 32768
fig, ax = plt.subplots(1, 4, figsize=(18, 2.6))
for a, (n, v) in zip(ax, [("original", _y), ("augmented #1", augment(_y, _rng)), ("augmented #2", augment(_y, _rng)), ("augmented #3", augment(_y, _rng))]):
    a.plot(v, lw=.4); a.set_title(n); a.set_xticks([])
plt.tight_layout(); show("fig03_augmentation_examples")

class WaveDataset(torch.utils.data.Dataset):
    """(waveform[48000], label). Augmentation only when augment_on=True (training split)."""
    def __init__(self, idx, augment_on=False):
        self.idx, self.augment_on = np.asarray(idx), augment_on
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        j = self.idx[k]
        y = WAVES[j].astype(np.float32) / 32768.0
        if self.augment_on:
            y = augment(y, np.random.default_rng())          # fresh randomness per call (safe with workers)
        return torch.from_numpy(y), int(LABELS[j])

def make_loader(idx, batch_size, train):
    return torch.utils.data.DataLoader(WaveDataset(idx, augment_on=train), batch_size=batch_size, shuffle=train,
                                       num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                                       persistent_workers=NUM_WORKERS > 0)

# %% [markdown]
# ## 6. Model architectures
#
# ### 6.1 Overall system
# The research part (this notebook) produces a model; the engineering part (§13) turns it into a service:
# `microphone → 16 kHz WAV → REST API (FastAPI + ONNX Runtime) → emotion + confidence → Android UI`.

# %%
def draw_pipeline():
    """Figure: end-to-end system diagram drawn with matplotlib (no external assets)."""
    fig, ax = plt.subplots(figsize=(16, 3.2)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 3)
    boxes = [("Microphone\n(Android app)", 0.2, "#F58518"), ("16 kHz mono\nWAV upload", 2.7, "#F58518"), ("Trim silence\n3 s windows", 5.2, "#4C78A8"),
             ("Fine-tuned SSL\ntransformer (ONNX)", 7.7, "#4C78A8"), ("Softmax over\n7 emotions", 10.2, "#4C78A8"), ("Average windows\n→ emotion + confidence", 12.7, "#54A24B")]
    for i, (t, x, c) in enumerate(boxes):
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.8), 1.95, 1.4, boxstyle="round,pad=0.05", fc=c, ec="none", alpha=.9))
        ax.text(x + 0.975, 1.5, t, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if i < len(boxes) - 1: ax.annotate("", xy=(x + 2.5, 1.5), xytext=(x + 2.0, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(8, 2.7, "End-to-end speech-emotion system", ha="center", fontsize=13, fontweight="bold")
    ax.text(3.8, 0.35, "client", ha="center", color="#F58518"); ax.text(9.0, 0.35, "server (free-tier cloud, CPU)", ha="center", color="#4C78A8")
    plt.tight_layout(); show("fig04_system_pipeline")
draw_pipeline()

# %% [markdown]
# ### 6.2 Model A — CNN + BiLSTM + Attention on log-mel (lightweight baseline)
# * **Front-end:** fixed (non-trainable) STFT → mel filter-bank → log → per-utterance standardisation; implemented with 1-D convolutions so that the *whole model including feature extraction* exports to ONNX.
# * **CNN (4 blocks of Conv-BN-ReLU-MaxPool-Dropout):** learns local time-frequency patterns (formant movements, pitch contours).
# * **BiLSTM:** reads the CNN frame sequence forwards and backwards, $h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]$, capturing how prosody evolves over the utterance.
# * **Additive attention pooling:** not every frame is equally emotional; the network learns where to look:
#   $$ e_t = v^\top\tanh(W h_t + b),\qquad \alpha_t=\frac{\exp e_t}{\sum_\tau \exp e_\tau},\qquad c=\sum_t \alpha_t h_t $$
# * **Classifier:** dropout → 64-unit dense → 7 logits. SpecAugment masks are applied in training only.
#
# ≈ 0.95 M trainable parameters (plus ≈ 1.1 M fixed spectrogram constants), 8 MB as ONNX: this is the model that comfortably fits any free hosting tier.

# %%
def utt_norm(x):
    """Per-utterance zero-mean / unit-variance normalisation (removes microphone-gain differences)."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-7)

class MelFrontEnd(nn.Module):
    """Log-mel via fixed conv1d-STFT (cos/sin kernels * Hann window) -> only conv/matmul ops, so it exports cleanly to ONNX."""
    def __init__(self, sr=SR, n_fft=1024, hop=256, n_mels=64, f_min=20., f_max=8000.):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        win = torch.hann_window(n_fft)
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float(); n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * math.pi * k * n / n_fft
        self.register_buffer("cos_k", (torch.cos(ang) * win).unsqueeze(1))        # (F,1,n_fft)
        self.register_buffer("sin_k", (-torch.sin(ang) * win).unsqueeze(1))
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sr, norm="slaney", mel_scale="slaney")
        self.register_buffer("mel_fb", fb)                                        # (F, n_mels)
    def forward(self, x):                                                         # x: (B, T)
        x = F.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re_, im_ = F.conv1d(x, self.cos_k, stride=self.hop), F.conv1d(x, self.sin_k, stride=self.hop)
        power = re_ ** 2 + im_ ** 2                                               # |STFT|^2   (B,F,frames)
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)    # (B, n_mels, frames)
        lm = torch.log(mel + 1e-6)
        lm = (lm - lm.mean(dim=(1, 2), keepdim=True)) / torch.sqrt(lm.var(dim=(1, 2), keepdim=True, unbiased=False) + 1e-5)
        return lm.unsqueeze(1)                                                    # (B, 1, n_mels, frames)

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_mels=64, hidden=128, dropout=0.3):
        super().__init__()
        self.frontend = MelFrontEnd(n_mels=n_mels)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10, iid_masks=True)   # SpecAugment
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20, iid_masks=True)
        def block(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(pool), nn.Dropout2d(0.1))
        self.cnn = nn.Sequential(block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 128, (2, 1)), block(128, 128, (2, 1)))
        self.lstm = nn.LSTM(128 * (n_mels // 16), hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1))            # additive attention
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
    def forward(self, wav):
        x = self.frontend(utt_norm(wav))                         # (B,1,M,T)
        if self.training:                                        # SpecAugment (training only)
            x = self.time_mask(self.freq_mask(x))
        x = self.cnn(x)                                          # (B,C,M/16,T/4)
        b, c, f, t = x.shape
        h, _ = self.lstm(x.permute(0, 3, 1, 2).reshape(b, t, c * f))   # (B,T',2H)
        w = torch.softmax(self.att(h), dim=1)                    # attention weights over time
        return self.head((w * h).sum(dim=1))
    def param_groups(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": LR_A}]

# %% [markdown]
# ### 6.3 Model B — fine-tuned self-supervised speech transformers
# **Background.** Wav2Vec 2.0, HuBERT and WavLM are transformers *pre-trained on tens of thousands of hours of unlabeled speech* with self-supervised objectives (e.g. Wav2Vec 2.0 solves a contrastive task: identify the true quantised latent among distractors,
# $\mathcal{L}_m=-\log\frac{\exp(\mathrm{sim}(c_t,q_t)/\kappa)}{\sum_{\tilde q}\exp(\mathrm{sim}(c_t,\tilde q)/\kappa)}$; HuBERT predicts offline cluster assignments of masked frames; WavLM adds denoising and gated relative position bias).
# They already encode phonetic *and* paralinguistic information, so a small labelled emotion corpus is enough to **fine-tune** them.
#
# **Our head.**
# 1. Raw waveform → CNN feature encoder (kept **frozen**) → 12 transformer layers (**fine-tuned**).
# 2. **Learned weighted sum of all 13 hidden states** (embedding output + 12 layers), as in the SUPERB benchmark:
#    $$ h=\sum_{l=0}^{12}\operatorname{softmax}(a)_l\,H_l $$
#    Lower layers hold acoustic/prosodic detail, upper layers hold linguistic content; emotion needs the former, so letting the model *choose* the mix is better than using only the last layer. We plot the learned weights in §11.
# 3. Mean-pool over time → LayerNorm → Linear(768→256) → ReLU → Dropout → Linear(256→7).
#
# **Why three backbones?** They share the architecture but differ in pre-training; comparing them tells us which pre-training objective suits emotion best. The one with the best *validation* accuracy gets the expensive Protocol 2 evaluation.

# %%
from transformers import AutoModel

class SSLClassifier(nn.Module):
    """Self-supervised speech transformer + learned layer-weighted pooling + MLP head. Input: raw waveform (B, 48000)."""
    def __init__(self, name, n_classes=N_CLASSES, attn=None):
        super().__init__()
        kw = dict(layerdrop=0.0, mask_time_prob=0.05, mask_time_length=5)     # light time-masking acts as regularisation
        if attn: kw["attn_implementation"] = attn                              # "eager" is the ONNX-friendly implementation
        self.backbone = AutoModel.from_pretrained(HF_IDS[name], **kw)
        for prm in self.backbone.feature_extractor.parameters():               # the low-level CNN feature encoder stays frozen
            prm.requires_grad = False                                          # (done by hand: not every backbone class offers freeze_feature_encoder)
        if hasattr(self.backbone.feature_extractor, "_requires_grad"): self.backbone.feature_extractor._requires_grad = False
        d, n_layers = self.backbone.config.hidden_size, self.backbone.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(n_layers))                     # a_l (softmax -> weights); starts uniform
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))
    def forward(self, wav):
        hs = self.backbone(utt_norm(wav), output_hidden_states=True).hidden_states     # tuple of (B,T',d)
        w = torch.softmax(self.layer_w, dim=0).view(-1, 1, 1, 1)
        h = (w * torch.stack(hs, dim=0)).sum(dim=0)                                    # weighted sum of layers
        return self.head(self.norm(h.mean(dim=1)))                                     # mean-pool over time
    def param_groups(self):
        back = [p for p in self.backbone.parameters() if p.requires_grad]
        rest = [self.layer_w] + list(self.norm.parameters()) + list(self.head.parameters())
        return [{"params": back, "lr": LR_B_BACKBONE}, {"params": rest, "lr": LR_B_HEAD}]

ARCH = {"CNN-BiLSTM-Attn": dict(build=lambda: CNNBiLSTMAttention(), amp=False, bs=BATCH_A, ep1=P1_EPOCHS_A, ep2=P2_EPOCHS_A, kind="cnn")}
for _n in HF_IDS:
    ARCH[_n] = dict(build=(lambda n=_n: SSLClassifier(n)), amp=True, bs=BATCH_B, ep1=P1_EPOCHS_B, ep2=P2_EPOCHS_B, kind="ssl")

# Model summary table (parameters, trainable parameters, output shape check)
summary = []
for n, a in ARCH.items():
    m = a["build"](); out = m.eval()(torch.randn(2, N_SAMPLES))
    summary.append({"Model": n, "Params (M)": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                    "Trainable (M)": round(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 1), "Output": tuple(out.shape)})
    del m
display(pd.DataFrame(summary).set_index("Model"))

# %% [markdown]
# ## 7. Training procedure
# One generic loop trains every model so the comparison is fair:
# * **Optimiser:** AdamW (decoupled weight decay 0.01) with **One-Cycle** learning-rate schedule (warm-up 15 % then cosine decay). Transformer backbones use a small LR ($5\!\times\!10^{-5}$) to avoid destroying pre-trained knowledge; the new head / layer weights use $10^{-3}$.
# * **Loss:** class-weighted cross-entropy with label smoothing 0.05 (§3).
# * **Stabilisers:** gradient clipping (‖g‖ ≤ 1), mixed precision (fp16) for transformers on GPU.
# * **Model selection:** after every epoch we compute **validation** accuracy and keep the best epoch's weights. The test set is *never* looked at during training or selection.

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
    """Train `arch` on train_idx, select the best epoch on val_idx. Returns (model with best weights, history DataFrame)."""
    cfg = ARCH[arch]; seed_everything()
    model = cfg["build"]().to(DEVICE)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=LABELS[train_idx])     # weights from the TRAIN fold only
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
        print(f"[{arch}{tag}] epoch {ep+1:2d}/{epochs}  loss {tl/n:.3f}  train_acc {correct/n:.3f}  val_acc {va:.3f}  ({time.time()-t0:.0f}s)")
        if va > best_acc:
            best_acc = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

def plot_cm(y, p, title, name):
    cm = confusion_matrix(y, p, labels=range(N_CLASSES), normalize="true")
    plt.figure(figsize=(5.6, 4.6)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS, vmin=0, vmax=1)
    plt.title(title); plt.ylabel("true emotion"); plt.xlabel("predicted emotion"); plt.tight_layout(); show(name)

def per_dataset_table(idx, pred):
    """Accuracy per source corpus for clips `idx` with predictions `pred` (explains the combined score)."""
    d = pd.DataFrame({"dataset": DATASETS[idx], "ok": (np.asarray(pred) == LABELS[idx])})
    t = d.groupby("dataset").ok.agg(["mean", "count"]); t.columns = ["accuracy", "n_clips"]
    return t

# %% [markdown]
# ## 8. Evaluation protocols & metrics
#
# We deliberately report **two numbers that answer two different questions** and never mix them.
#
# | | **Protocol 1 — Stratified random split** | **Protocol 2 — Speaker-independent GroupKFold** |
# |---|---|---|
# | Split | 80 / 10 / 10 train / val / test, stratified by emotion, shuffled over *all* speakers | 5 folds; **groups = speaker ID**; every speaker is in exactly one test fold |
# | Question answered | “How well does the model do on *new recordings* of speakers it has heard?” | “How well does it do on a **person it has never heard** (= a real user)?” |
# | Leakage | the same speakers (and sessions, microphones, sentences) occur in train *and* test | none: speaker sets are disjoint (asserted in code) |
# | Expected value | high (headline number) | lower, **honest real-world estimate** |
# | Used for | comparison with the literature | choosing the deployed model |
#
# **Metrics.** Accuracy $=\frac1N\sum_i \mathbb{1}[\hat y_i=y_i]$ and **macro-F1** $=\frac1K\sum_c F1_c$ (treats rare classes such as *surprise* equally). For Protocol 2 we report the **mean ± standard deviation over folds** and a pooled out-of-fold confusion matrix.
# For Protocol 2 the validation set used for epoch selection is made of **held-out speakers carved out of the training fold**, so the test fold is touched exactly once.

# %%
all_idx = np.arange(len(df))
tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=LABELS, random_state=SEED, shuffle=True)
va_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=LABELS[tmp_idx], random_state=SEED, shuffle=True)
shared = set(SPEAKERS[te_idx]) & set(SPEAKERS[tr_idx])
print(f"Protocol 1 split: train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")
print(f"Leakage check: {len(shared)} of {len(set(SPEAKERS[te_idx]))} test speakers also appear in training (expected -> NOT speaker-independent)")

# %% [markdown]
# ## 9. Experiment 1 — Stratified random split (Protocol 1)
# All four models are trained on the same split. For each we store validation/test probabilities and the training history so every figure in the report can be regenerated without retraining.

# %%
def run_protocol1(arch):
    if arch in RESULTS["p1"]:
        print(f"[{arch}] Protocol 1 already done (test acc {RESULTS['p1'][arch]['acc']*100:.2f}%) - skipping"); return
    model, hist = train_model(arch, tr_idx, va_idx, ARCH[arch]["ep1"], tag=" P1")
    amp = ARCH[arch]["amp"]
    tp, vp = predict_probs(model, te_idx, amp=amp), predict_probs(model, va_idx, amp=amp)
    pred = tp.argmax(1)
    RESULTS["p1"][arch] = dict(acc=float(accuracy_score(LABELS[te_idx], pred)), macro_f1=float(f1_score(LABELS[te_idx], pred, average="macro")),
                               val_acc=float(accuracy_score(LABELS[va_idx], vp.argmax(1))), n_test=len(te_idx), pred=pred.tolist(),
                               test_probs=tp.round(5).tolist(), val_probs=vp.round(5).tolist(), hist=hist.to_dict("records"))
    save_results()
    print(f"\n=== {arch} | Protocol 1 test accuracy {RESULTS['p1'][arch]['acc']*100:.2f}% | macro-F1 {RESULTS['p1'][arch]['macro_f1']*100:.2f}% ===\n")
    del model; torch.cuda.empty_cache()

run_protocol1("CNN-BiLSTM-Attn")

# %%
run_protocol1("Wav2Vec2-base")

# %%
run_protocol1("HuBERT-base")

# %%
run_protocol1("WavLM-base-plus")

# %% [markdown]
# ### 9.1 Optional ensemble
# Averaging the probabilities of several models usually helps because their errors are only partly correlated. The subset of models is chosen **on validation accuracy only**. It is disabled by default (`ENABLE_ENSEMBLE = False`) because a deployed system should be a single small model; we include it to show the attainable upper bound.

# %%
if ENABLE_ENSEMBLE and sum(a in RESULTS["p1"] for a in HF_IDS) >= 2:
    names = [a for a in HF_IDS if a in RESULTS["p1"]]
    best = max((c for r in range(2, len(names) + 1) for c in itertools.combinations(names, r)),
               key=lambda c: accuracy_score(LABELS[va_idx], np.mean([np.array(RESULTS["p1"][a]["val_probs"]) for a in c], 0).argmax(1)))
    tp = np.mean([np.array(RESULTS["p1"][a]["test_probs"]) for a in best], 0); pred = tp.argmax(1)
    RESULTS["p1"]["Ensemble (" + "+".join(best) + ")"] = dict(acc=float(accuracy_score(LABELS[te_idx], pred)),
        macro_f1=float(f1_score(LABELS[te_idx], pred, average="macro")), n_test=len(te_idx), pred=pred.tolist(), members=list(best))
    save_results(); print("Ensemble members:", best, "| test acc %.2f%%" % (RESULTS["p1"][list(RESULTS["p1"])[-1]]["acc"] * 100))
else:
    print("Ensemble disabled (set ENABLE_ENSEMBLE = True to evaluate it).")

# %% [markdown]
# ### 9.2 Training dynamics
# Training curves reveal under-/over-fitting. A growing gap between training and validation accuracy indicates over-fitting; curves that are still rising at the last epoch indicate that more training would help.

# %%
p1_models = [a for a in ARCH if a in RESULTS["p1"]]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a in p1_models:
    h = pd.DataFrame(RESULTS["p1"][a]["hist"])
    ax[0].plot(h.epoch, h.train_loss, label=a); ax[1].plot(h.epoch, h.val_acc * 100, label=a)
ax[0].set_title("Training loss (class-weighted, label-smoothed)"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].set_title("Validation accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); show("fig05_training_curves")

# %%
# Protocol 1 results: table + confusion matrices + per-dataset breakdown
rows = []
for a in p1_models:
    r = RESULTS["p1"][a]; rows.append({"Model": a, "Val acc (%)": round(r.get("val_acc", np.nan) * 100, 2),
                                       "Test acc (%)": round(r["acc"] * 100, 2), "Macro-F1 (%)": round(r["macro_f1"] * 100, 2)})
p1_table = pd.DataFrame(rows).set_index("Model"); display(p1_table)
for a in p1_models:
    r = RESULTS["p1"][a]
    print(f"\n##### {a} — Protocol 1 (stratified random split) #####")
    print(classification_report(LABELS[te_idx], r["pred"], labels=range(N_CLASSES), target_names=EMOTIONS, digits=3, zero_division=0))
    plot_cm(LABELS[te_idx], r["pred"], f"{a} — Protocol 1", f"fig06_cm_p1_{a}")

# %%
ds_rows = {a: per_dataset_table(te_idx, RESULTS["p1"][a]["pred"]).accuracy * 100 for a in p1_models}
p1_by_ds = pd.DataFrame(ds_rows).round(1); display(p1_by_ds.assign(n_test_clips=per_dataset_table(te_idx, RESULTS["p1"][p1_models[0]]["pred"]).n_clips))
p1_by_ds.plot.bar(figsize=(10, 4), rot=0); plt.ylabel("accuracy (%)"); plt.ylim(0, 105); plt.title("Protocol 1 accuracy per source dataset")
plt.axhline(95, ls="--", c="gray", lw=.8); plt.tight_layout(); show("fig07_p1_per_dataset")

# %% [markdown]
# **Reading Experiment 1.** Clean, low-speaker-count corpora (TESS, SAVEE, RAVDESS) are typically recognised almost perfectly when speakers are shared between train and test, whereas CREMA-D — crowd-labelled, with ambiguous expressions — is much harder and is ≈ 60 % of the test set.
# The combined random-split accuracy is therefore bounded by CREMA-D difficulty; see the per-dataset table above and the discussion in §11.

# %% [markdown]
# ## 10. Experiment 2 — Speaker-independent 5-fold GroupKFold (Protocol 2)
# Protocol 2 is repeated 5 times per model, so we run it for the CNN baseline and for the **best transformer by Protocol-1 validation accuracy** (chosen without touching the test set). For each fold the best-epoch model is evaluated *once* on the unseen speakers; the best fold's weights are kept for deployment.

# %%
def run_protocol2(arch):
    res = RESULTS["p2"].setdefault(arch, {"folds": []})
    done = {f["fold"] for f in res["folds"]}
    for k, (trn, tst) in enumerate(GroupKFold(n_splits=N_FOLDS).split(all_idx, LABELS, groups=SPEAKERS)):
        if k in done:
            print(f"[{arch}] fold {k} already done"); continue
        assert not (set(SPEAKERS[trn]) & set(SPEAKERS[tst])), "speaker leakage between train and test!"
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(trn, groups=SPEAKERS[trn]))   # inner val = unseen speakers
        model, hist = train_model(arch, trn[a], trn[b], ARCH[arch]["ep2"], tag=f" P2 fold{k}")
        pred = predict_probs(model, tst, amp=ARCH[arch]["amp"]).argmax(1)
        acc = float(accuracy_score(LABELS[tst], pred))
        res["folds"].append(dict(fold=k, acc=acc, macro_f1=float(f1_score(LABELS[tst], pred, average="macro")), n_test=len(tst),
                                 n_test_speakers=len(set(SPEAKERS[tst])), idx=tst.tolist(), pred=pred.tolist(), hist=hist.to_dict("records")))
        if acc >= max(f["acc"] for f in res["folds"]):          # keep the best fold's weights for deployment
            torch.save(model.state_dict(), os.path.join(WORK_DIR, f"best_p2_{arch}_{TAG}.pt")); res["best_fold"] = k
        save_results()
        print(f"--- {arch} | fold {k}: unseen-speaker accuracy {acc*100:.2f}% ({len(set(SPEAKERS[tst]))} test speakers) ---")
        del model; torch.cuda.empty_cache()
    accs = [f["acc"] for f in res["folds"]]
    res["mean"], res["std"] = float(np.mean(accs)), float(np.std(accs)); save_results()
    print(f"\n=== {arch} | Protocol 2 ({N_FOLDS}-fold, unseen speakers): {res['mean']*100:.2f}% ± {res['std']*100:.2f}% ===\n")

run_protocol2("CNN-BiLSTM-Attn")

# %%
# Choose the transformer(s) for Protocol 2 using VALIDATION accuracy from Protocol 1 (never the test set)
ssl_ranked = sorted([a for a in HF_IDS if a in RESULTS["p1"]], key=lambda a: -RESULTS["p1"][a]["val_acc"])
P2_CHOSEN = ssl_ranked[:P2_TRANSFORMERS]
print("Transformers ranked by P1 validation accuracy:", [(a, round(RESULTS["p1"][a]["val_acc"] * 100, 2)) for a in ssl_ranked])
print("-> Protocol 2 will be run for:", P2_CHOSEN)
for a in P2_CHOSEN:
    run_protocol2(a)

# %%
p2_models = [a for a in ARCH if "mean" in RESULTS["p2"].get(a, {})]
for a in p2_models:
    r = RESULTS["p2"][a]
    print(f"\n##### {a} — Protocol 2 (speaker-independent) #####")
    print(pd.DataFrame([{k: v for k, v in f.items() if k in ("fold", "acc", "macro_f1", "n_test", "n_test_speakers")} for f in r["folds"]]).round(4).to_string(index=False))
    idx = np.concatenate([f["idx"] for f in r["folds"]]); pred = np.concatenate([f["pred"] for f in r["folds"]])
    print(classification_report(LABELS[idx], pred, labels=range(N_CLASSES), target_names=EMOTIONS, digits=3, zero_division=0))
    plot_cm(LABELS[idx], pred, f"{a} — Protocol 2 (pooled out-of-fold)", f"fig08_cm_p2_{a}")
    r["pooled_idx"], r["pooled_pred"] = idx.tolist(), pred.tolist()

# %%
p2_by_ds = pd.DataFrame({a: per_dataset_table(np.array(RESULTS["p2"][a]["pooled_idx"]), RESULTS["p2"][a]["pooled_pred"]).accuracy * 100 for a in p2_models}).round(1)
display(p2_by_ds)
p2_by_ds.plot.bar(figsize=(10, 4), rot=0); plt.ylabel("accuracy (%)"); plt.ylim(0, 105); plt.title("Protocol 2 (unseen speakers) accuracy per source dataset")
plt.tight_layout(); show("fig09_p2_per_dataset")

# %% [markdown]
# ## 11. Analysis & discussion

# %% [markdown]
# ### 11.1 Both protocols side by side
# The table and bar chart place the two protocols next to each other. **They must not be conflated:** the random-split bar is what a model achieves when its test speakers also appear in training; the speaker-independent bar is the estimate for a brand-new user.

# %%
rows = []
for a in ARCH:
    p1, p2 = RESULTS["p1"].get(a), RESULTS["p2"].get(a, {})
    rows.append({"Model": a, "P1 random-split test acc (%)": round(p1["acc"] * 100, 2) if p1 else np.nan,
                 "P2 speaker-independent mean acc (%)": round(p2["mean"] * 100, 2) if "mean" in p2 else np.nan,
                 "P2 std (%)": round(p2["std"] * 100, 2) if "std" in p2 else np.nan})
cmp_df = pd.DataFrame(rows); cmp_df["Gap P1 − P2 (pts)"] = (cmp_df.iloc[:, 1] - cmp_df.iloc[:, 2]).round(2)
display(cmp_df.set_index("Model"))

fig, ax = plt.subplots(figsize=(10, 4.8)); x = np.arange(len(cmp_df)); w = 0.38
b1 = ax.bar(x - w/2, cmp_df.iloc[:, 1], w, label="Protocol 1: stratified random split (speakers shared)", color="#4C78A8")
b2 = ax.bar(x + w/2, cmp_df.iloc[:, 2], w, yerr=cmp_df["P2 std (%)"].fillna(0), capsize=4, label="Protocol 2: speaker-independent GroupKFold (unseen speakers)", color="#F58518")
for b in list(b1) + list(b2):
    if not np.isnan(b.get_height()): ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cmp_df.Model, rotation=10); ax.set_ylim(0, 108); ax.set_ylabel("accuracy (%)")
ax.axhline(95, ls="--", c="gray", lw=.8); ax.set_title("Two evaluation protocols — never to be conflated"); ax.legend(loc="lower center", fontsize=8)
plt.tight_layout(); show("fig10_protocol_comparison")

# %% [markdown]
# ### 11.2 Why is Protocol 2 lower? (and why that is *good* science)
# * **Speaker / session leakage in Protocol 1.** Each corpus contains many near-identical recordings per speaker (same sentences, same studio, same microphone). With a random split the test set contains *the same speakers* as training, so a flexible model can partly recognise *who is speaking and how the room sounds* rather than purely *what emotion is expressed*.
# * **Protocol 2 removes the shortcut.** Disjoint speakers mimic deployment: a new person, a new phone.
# * **Small speaker pools.** TESS (2 speakers) and SAVEE (4) give little diversity; each fold removes a large fraction of a speaker's data.
# * **Label noise in CREMA-D.** Its labels come from crowd ratings and human agreement is only ≈ 40–60 %, which caps the achievable accuracy on ~60 % of the data.
# * **Acted vs. natural speech.** Actors exaggerate emotions; natural speech is subtler, so real-world accuracy can be lower still.
# A gap between the protocols is therefore *expected*; identical scores would suggest the data were too easy or the evaluation leaky.

# %%
# Error analysis: the most frequent confusions of the best model under Protocol 2 (or Protocol 1 if P2 is unavailable)
ref = max(p2_models, key=lambda a: RESULTS["p2"][a]["mean"]) if p2_models else p1_models[0]
if p2_models:
    yt, yp = LABELS[np.array(RESULTS["p2"][ref]["pooled_idx"])], np.array(RESULTS["p2"][ref]["pooled_pred"]); proto = "Protocol 2"
else:
    yt, yp = LABELS[te_idx], np.array(RESULTS["p1"][ref]["pred"]); proto = "Protocol 1"
cm = confusion_matrix(yt, yp, labels=range(N_CLASSES)); off = [(EMOTIONS[i], EMOTIONS[j], cm[i, j], cm[i, j] / cm[i].sum()) for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j]
conf_df = pd.DataFrame(sorted(off, key=lambda r: -r[2])[:8], columns=["true", "predicted as", "count", "share of true class"]).round(3)
print(f"Most frequent confusions — {ref}, {proto}:"); display(conf_df)

# %% [markdown]
# *Interpretation.* Confusions concentrate on emotions that share prosodic cues — typically *neutral* ↔ *sad*, *fear* ↔ *sad*, *angry* ↔ *disgust/happy* (high-arousal pairs). This agrees with the psychology literature, where the *arousal* dimension (calm ↔ excited) is recognised far more reliably than *valence* (positive ↔ negative).

# %%
# Which transformer layers does the best model use? (learned weights of the weighted layer sum)
ssl_p2 = [a for a in p2_models if a in HF_IDS]
if ssl_p2:
    a = ssl_p2[0]
    state = torch.load(os.path.join(WORK_DIR, f"best_p2_{a}_{TAG}.pt"), map_location="cpu")
    lw = torch.softmax(state["layer_w"], 0).numpy()
    plt.figure(figsize=(8, 3.5)); plt.bar(range(len(lw)), lw, color="#4C78A8"); plt.xlabel("hidden state (0 = CNN/embedding output, 1–12 = transformer layers)")
    plt.ylabel("softmax weight"); plt.title(f"{a}: learned layer weights"); plt.tight_layout(); show("fig11_layer_weights")
    print("Most used layers:", np.argsort(-lw)[:3].tolist())

# %% [markdown]
# *Interpretation.* Weight mass on lower/middle layers (rather than only the top layer) confirms that emotion cues are mainly **acoustic/prosodic**, which is why a plain “last-layer” head is sub-optimal and the learned layer mix is worthwhile.

# %% [markdown]
# ## 12. Final results (auto-generated from the measured numbers)
# The cell below builds the summary table and a **ready-to-paste abstract / conclusion paragraph** with the actual values. Nothing here is hard-coded.

# %%
from IPython.display import Markdown, display
lines = ["### Final results\n", "| Model | Random split (P1) | Speaker-independent (P2) |", "|---|---|---|"]
for a in ARCH:
    p1, p2 = RESULTS["p1"].get(a), RESULTS["p2"].get(a, {})
    lines.append(f"| {a} | {p1['acc']*100:.1f} % (F1 {p1['macro_f1']*100:.1f} %) | " +
                 (f"{p2['mean']*100:.1f} ± {p2['std']*100:.1f} % |" if "mean" in p2 else "not run |") if p1 else f"| {a} | – | – |")
best_p1 = max(p1_models, key=lambda a: RESULTS["p1"][a]["acc"])
best_p2 = max(p2_models, key=lambda a: RESULTS["p2"][a]["mean"]) if p2_models else None
summary_md = "\n".join(lines) + f"\n\n**Best random-split model:** {best_p1} — **{RESULTS['p1'][best_p1]['acc']*100:.1f} %** test accuracy " \
    f"({'≥' if RESULTS['p1'][best_p1]['acc'] >= .95 else '<'} the 95 % target).  "
if best_p2:
    summary_md += f"**Best speaker-independent model:** {best_p2} — **{RESULTS['p2'][best_p2]['mean']*100:.1f} % ± {RESULTS['p2'][best_p2]['std']*100:.1f} %** over {N_FOLDS} folds of unseen speakers."
summary_md += ("\n\n**Per-dataset (best random-split model):** " + ", ".join(f"{d} {v:.1f} %" for d, v in (per_dataset_table(te_idx, RESULTS['p1'][best_p1]['pred']).accuracy * 100).items()))
display(Markdown(summary_md))

abstract = (f"We built a 7-class speech emotion recogniser from four public corpora ({len(df):,} utterances, {df.speaker.nunique()} speakers). "
    f"The best model under a stratified random split ({best_p1}) reaches {RESULTS['p1'][best_p1]['acc']*100:.1f} % test accuracy, "
    + (f"whereas the speaker-independent {N_FOLDS}-fold GroupKFold accuracy of {best_p2} is {RESULTS['p2'][best_p2]['mean']*100:.1f} % ± {RESULTS['p2'][best_p2]['std']*100:.1f} %. " if best_p2 else "")
    + "The random-split figure reflects speakers shared between training and testing; the speaker-independent figure is the honest estimate of real-world performance. "
    "The deployed model is exported to ONNX and served through a REST API to an Android application.")
print("\n=== READY-TO-PASTE ABSTRACT ===\n" + abstract)
open(os.path.join(WORK_DIR, "report_summary.md"), "w").write(summary_md + "\n\n" + abstract)

# %% [markdown]
# ## 13. Deployment — from research model to a real-time service
# ### 13.1 Export to ONNX and quantisation
# The speaker-independent winner (highest Protocol-2 mean accuracy) is exported with `torch.onnx.export` (input `waveform: (batch, 48000)` → `logits: (batch, 7)`, dynamic batch axis).
# **ONNX Runtime** executes the graph on CPU without PyTorch (a much smaller runtime), and **8-bit dynamic quantisation** stores the weights as integers,
# $x_q=\mathrm{round}(x/s)+z$, to shrink the file ≈ 4× and speed up matrix multiplications.
# We quantise **only the MatMul layers** (the transformer's linear maps) and keep the convolutional feature encoder in fp32 because it is very sensitive to quantisation noise. We then **measure** the accuracy loss on real held-out clips instead of assuming it is negligible, and recommend INT8 only if the loss is small.

# %%
assert p2_models, "Run Protocol 2 first"
exp_candidates = sorted(p2_models, key=lambda a: -RESULTS["p2"][a]["mean"])
print("Export candidates (best speaker-independent first):", [(a, round(RESULTS["p2"][a]["mean"] * 100, 1)) for a in exp_candidates])

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
EXPORT_DIR = os.path.join(WORK_DIR, "export"); os.makedirs(EXPORT_DIR, exist_ok=True)
ONNX_PATH, INT8_PATH = os.path.join(EXPORT_DIR, "model.onnx"), os.path.join(EXPORT_DIR, "model_int8.onnx")

def build_for_export(arch):
    return CNNBiLSTMAttention() if ARCH[arch]["kind"] == "cnn" else SSLClassifier(arch, attn="eager")

EXPORTED = None
for arch in exp_candidates:                       # fall back to the next candidate if an export fails
    try:
        m = build_for_export(arch); m.load_state_dict(torch.load(os.path.join(WORK_DIR, f"best_p2_{arch}_{TAG}.pt"), map_location="cpu")); m.eval().cpu()
        kw = dict(input_names=["waveform"], output_names=["logits"], dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
        try:    torch.onnx.export(m, (torch.randn(1, N_SAMPLES),), ONNX_PATH, dynamo=False, **kw)
        except TypeError: torch.onnx.export(m, (torch.randn(1, N_SAMPLES),), ONNX_PATH, **kw)
        EXPORTED = arch; EXPORT_MODEL = m; print("exported:", arch, f"({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)"); break
    except Exception as e:
        print(f"export of {arch} failed ({repr(e)[:160]}) -> trying next candidate")
assert EXPORTED, "no model could be exported"

try:
    quantize_dynamic(ONNX_PATH, INT8_PATH, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=True)
except Exception as e:
    print("INT8 quantisation failed:", repr(e)[:200]); INT8_PATH = None

# %% [markdown]
# ### 13.2 Fidelity, size and latency of the exported models
# We compare PyTorch, ONNX-fp32 and ONNX-INT8 on **real clips from the best fold's unseen-speaker test set** (up to 300 clips): accuracy, agreement with the fp32 model, file size and single-thread CPU latency for one 3 s window (a proxy for a small free-tier server).

# %%
best_fold = RESULTS["p2"][EXPORTED]["best_fold"]
fold_rec = next(f for f in RESULTS["p2"][EXPORTED]["folds"] if f["fold"] == best_fold)
chk_idx = np.array(fold_rec["idx"])[:300]
xb = np.stack([WAVES[i].astype(np.float32) / 32768 for i in chk_idx]); yb = LABELS[chk_idx]

def run_onnx(path, x, bs=8, threads=2):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    out = np.concatenate([s.run(None, {"waveform": x[i:i + bs]})[0] for i in range(0, len(x), bs)])
    s1 = ort.SessionOptions(); s1.intra_op_num_threads = 1; s1 = ort.InferenceSession(path, s1, providers=["CPUExecutionProvider"])
    s1.run(None, {"waveform": x[:1]}); t0 = time.time(); [s1.run(None, {"waveform": x[:1]}) for _ in range(3)]
    return out, (time.time() - t0) / 3 * 1000

with torch.no_grad(): ref_logits = np.concatenate([EXPORT_MODEL(torch.from_numpy(xb[i:i + 8])).numpy() for i in range(0, len(xb), 8)])
fid = [{"Variant": "PyTorch fp32", "Accuracy (%)": round((ref_logits.argmax(1) == yb).mean() * 100, 2), "Agreement with PyTorch (%)": 100.0, "Size (MB)": np.nan, "Latency 1 thread (ms)": np.nan}]
for name, path in [("ONNX fp32", ONNX_PATH), ("ONNX int8 (MatMul only)", INT8_PATH)]:
    if not path: continue
    out, lat = run_onnx(path, xb)
    fid.append({"Variant": name, "Accuracy (%)": round((out.argmax(1) == yb).mean() * 100, 2), "Agreement with PyTorch (%)": round((out.argmax(1) == ref_logits.argmax(1)).mean() * 100, 2),
                "Size (MB)": round(os.path.getsize(path) / 1e6, 1), "Latency 1 thread (ms)": round(lat)})
fid_df = pd.DataFrame(fid).set_index("Variant"); display(fid_df)

# Recommendation rule: use INT8 if it loses < 1 accuracy point vs fp32, otherwise stay with fp32
use_int8 = INT8_PATH is not None and "ONNX int8 (MatMul only)" in fid_df.index and \
           fid_df.loc["ONNX fp32", "Accuracy (%)"] - fid_df.loc["ONNX int8 (MatMul only)", "Accuracy (%)"] < 1.0
SERVE_PATH = INT8_PATH if use_int8 else ONNX_PATH
print("Recommended serving model:", os.path.basename(SERVE_PATH))
json.dump({"labels": EMOTIONS, "sample_rate": SR, "window_samples": N_SAMPLES, "top_db": TOP_DB, "architecture": EXPORTED,
           "protocol2_mean_acc": RESULTS["p2"][EXPORTED]["mean"], "protocol1_acc": RESULTS["p1"].get(EXPORTED, {}).get("acc"),
           "recommended_file": os.path.basename(SERVE_PATH)}, open(os.path.join(EXPORT_DIR, "labels.json"), "w"), indent=2)

# %% [markdown]
# ### 13.3 Real-time inference with voice-activity detection
# For streaming audio we (1) run **silero-VAD** to find speech regions, (2) slide a **3 s window with 50 % overlap** (hop 1.5 s) over the signal, (3) classify only windows with enough speech, and (4) show *emotion + confidence* per window. The per-window latency below is measured on CPU.

# %%
from silero_vad import load_silero_vad, get_speech_timestamps
VAD = load_silero_vad()
SESS = ort.InferenceSession(SERVE_PATH, providers=["CPUExecutionProvider"])

def classify_window(y):
    """Raw mono 16 kHz audio (<= 3 s) -> (emotion, confidence, probabilities); same pre-processing as training."""
    logits = SESS.run(None, {"waveform": to_fixed(y)[None]})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return EMOTIONS[int(p.argmax())], float(p.max()), p

def stream_emotions(y, win_s=3.0, hop_s=1.5, min_speech=0.3, verbose=True):
    """Sliding-window emotion tracking with VAD gating. Returns one dict per window."""
    ts = get_speech_timestamps(torch.from_numpy(y), VAD, sampling_rate=SR)
    speech = np.zeros(len(y), dtype=bool)
    for t in ts: speech[t["start"]:t["end"]] = True
    W, H, out = int(win_s * SR), int(hop_s * SR), []
    for s in range(0, max(1, len(y) - W // 2), H):
        seg, sp = y[s:s + W], speech[s:s + W]
        rec = {"t_start": s / SR, "t_end": min(s + W, len(y)) / SR, "speech_frac": float(sp.mean()) if len(sp) else 0.0, "emotion": None, "confidence": None, "latency_ms": 0.0}
        if rec["speech_frac"] >= min_speech:
            t0 = time.time(); e, c, _ = classify_window(seg); rec.update(emotion=e, confidence=c, latency_ms=(time.time() - t0) * 1000)
        out.append(rec)
        if verbose:
            print(f"[{rec['t_start']:5.1f}–{rec['t_end']:5.1f}s] " + (f"{rec['emotion']:9s} {rec['confidence']*100:5.1f}%  ({rec['latency_ms']:.0f} ms)" if rec["emotion"] else "— no speech —"))
    return out

# Demo: concatenate unseen-speaker test clips of different emotions with 1 s of silence between them
rng = np.random.default_rng(1); parts, truth = [], []
for e in ["angry", "happy", "sad", "neutral", "fear"]:
    cand = [i for i in fold_rec["idx"] if LABELS[i] == LABEL2ID[e]]
    if cand: parts += [WAVES[rng.choice(cand)].astype(np.float32) / 32768, np.zeros(SR, np.float32)]; truth.append(e)
stream = np.concatenate(parts); print("ground-truth order:", truth)
res = stream_emotions(stream)

fig, ax = plt.subplots(figsize=(13, 3)); ax.plot(np.arange(len(stream)) / SR, stream, lw=.3, c="gray")
pal = sns.color_palette("tab10", N_CLASSES)
for r in res:
    if r["emotion"]:
        ax.axvspan(r["t_start"], r["t_end"], alpha=.12 + .25 * r["confidence"], color=pal[EMOTIONS.index(r["emotion"])])
        ax.text((r["t_start"] + r["t_end"]) / 2, 0.8 * np.abs(stream).max(), f"{r['emotion']}\n{r['confidence']*100:.0f}%", ha="center", fontsize=8)
ax.set_xlabel("time (s)"); ax.set_title("Sliding-window (3 s, 50 % overlap) emotion stream with VAD gating"); plt.tight_layout(); show("fig12_realtime_stream")

# %% [markdown]
# ### 13.4 Serving architecture and Android client
# **REST API (FastAPI + ONNX Runtime, hosted on a free Render web service).**
# `POST /predict` (multipart field `file` = WAV) → JSON
# ```json
# {"emotion": "happy", "confidence": 0.81, "probabilities": {"angry": 0.02, "...": 0.0, "surprise": 0.02},
#  "windows": [{"start_s": 0.0, "speech": true, "emotion": "happy", "confidence": 0.81}], "speech_detected": true, "latency_ms": 7000}
# ```
# The server decodes audio with libsndfile, resamples to 16 kHz (soxr), applies the *same* silence trimming as `to_fixed` (re-implemented in NumPy to avoid a heavy librosa dependency), scores windows one at a time (flat memory on a 512 MB free instance) and averages them.
#
# **Android app (Kotlin + Jetpack Compose).** Records up to 15 s with `AudioRecord` (16 kHz mono PCM → WAV), uploads it with OkHttp and shows the result: detected emotion, confidence, a short **wellness message with three tips**, per-class bars and an emotion-over-time line for longer recordings. User ID / password accounts are stored on the phone, and the server address lives in a Settings screen (free servers sleep when idle, so the app shows a server-status line and waits/retries).
#
# **Engineering trade-offs worth discussing in the report**
# | Constraint | Consequence | Mitigation |
# |---|---|---|
# | Free tier: 512 MB RAM, fractional CPU | large fp32 transformer does not fit / is slow | INT8 quantisation, one window per inference call, window cap |
# | Free tier sleeps after 15 min idle | first request takes ~1 min | status pill + retry, optional keep-alive pings |
# | Acted-speech training data | real-world accuracy lower | report speaker-independent accuracy; future: fine-tune on natural speech |

# %% [markdown]
# ## 14. Conclusion, limitations and future work
# **Conclusion.** We built, evaluated and deployed a 7-class speech emotion recogniser. Fine-tuned self-supervised transformers clearly outperform the spectrogram CNN-BiLSTM baseline, and a learned layer-weighted pooling lets the network use the prosody-rich intermediate layers. The most important methodological point is the **two-protocol evaluation**: the random-split number is the conventional headline, whereas the speaker-independent number is the honest estimate for real users. The final numbers are in §12.
#
# **Limitations**
# * Acted, mostly North-American English corpora — limited cultural/linguistic coverage; real conversations are subtler.
# * CREMA-D label noise caps achievable accuracy; *surprise* is under-represented and absent from CREMA-D.
# * Per-window 3 s analysis ignores long-range context; the free-tier server has multi-second latency.
# * 7 discrete categories oversimplify emotion (dimensional arousal/valence models are an alternative).
#
# **Future work.** (1) add more and more natural corpora (e.g. EMO-DB, MELD, IEMOCAP, MSP-Podcast); (2) larger backbones (wav2vec2-large, WavLM-large) with knowledge distillation into a mobile-sized student; (3) on-device inference with ONNX Runtime Mobile to remove the server; (4) multimodal fusion with text; (5) calibration and uncertainty estimation; (6) test-time augmentation and multi-seed ensembling.
#
# **References**
# 1. A. Baevski, Y. Zhou, A. Mohamed, M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” *NeurIPS*, 2020.
# 2. W.-N. Hsu et al., “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” *IEEE/ACM TASLP*, 2021.
# 3. S. Chen et al., “WavLM: Large-scale self-supervised pre-training for full-stack speech processing,” *IEEE JSTSP*, 2022.
# 4. S.-W. Yang et al., “SUPERB: Speech processing Universal PERformance Benchmark,” *Interspeech*, 2021.
# 5. S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS),” *PLoS ONE*, 2018.
# 6. H. Cao et al., “CREMA-D: Crowd-sourced emotional multimodal actors dataset,” *IEEE Trans. Affective Computing*, 2014.
# 7. M. K. Pichora-Fuller and K. Dupuis, “Toronto Emotional Speech Set (TESS),” Borealis Dataverse, 2020.
# 8. P. Jackson and S. Haq, “Surrey Audio-Visual Expressed Emotion (SAVEE) database,” University of Surrey, 2014.
# 9. D. S. Park et al., “SpecAugment: A simple data augmentation method for automatic speech recognition,” *Interspeech*, 2019.
# 10. L. Pepino, P. Riera, L. Ferrer, “Emotion recognition from speech using wav2vec 2.0 embeddings,” *Interspeech*, 2021.
# 11. Silero Team, “Silero VAD: pre-trained enterprise-grade voice activity detector,” 2021. · ONNX Runtime developers, “ONNX Runtime,” Microsoft, 2019.
#
# *(Please re-check the formatting/venue details of the references against your college's citation style before submission.)*

# %%
# Bundle every report artefact (figures + results + summary + deployment files) into one downloadable zip
shutil.copy(RESULTS_PATH, os.path.join(WORK_DIR, "results_for_report.json"))
OUT_DIR = "/kaggle/working/report_bundle" if IN_KAGGLE else os.path.join(WORK_DIR, "report_bundle")
os.makedirs(OUT_DIR, exist_ok=True)
for f in [ONNX_PATH, INT8_PATH, os.path.join(EXPORT_DIR, "labels.json"), os.path.join(WORK_DIR, "report_summary.md"), os.path.join(WORK_DIR, "results_for_report.json")]:
    if f and os.path.exists(f): shutil.copy(f, OUT_DIR)
shutil.copytree(FIG_DIR, os.path.join(OUT_DIR, "figures"), dirs_exist_ok=True)
zip_path = shutil.make_archive(os.path.join(os.path.dirname(OUT_DIR), "ser_report_bundle"), "zip", OUT_DIR)
print("Bundle ready:", zip_path, f"({os.path.getsize(zip_path)/1e6:.0f} MB)\nContains:", sorted(os.listdir(OUT_DIR)))
print("Deploy: copy", os.path.basename(SERVE_PATH), "as backend/model/model.onnx and labels.json next to it.")
