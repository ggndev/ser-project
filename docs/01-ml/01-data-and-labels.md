# 1a. Configuration, datasets and labels (notebook §2–3)

## Purpose
Collect all settings in one place, find the four datasets, read each file name to get **the emotion label and the speaker**, and make all datasets share one label set.

## Why this component

| Piece | What it does | Why we need it | Rejected alternative |
|---|---|---|---|
| Single configuration cell | All constants (sample rate, epochs, learning rates …) at the top | Reproducible; easy to change one number | Magic numbers spread over many cells |
| `find_root` | Finds a dataset folder on Kaggle or locally | Kaggle mounts datasets at different paths than Colab | Hard-coded paths break on another machine |
| File-name parsing | The emotion and speaker are encoded in file names | The datasets have no label files | Manual labelling (impossible for 12 000 clips) |
| Speaker ID per clip | Needed by Protocol 2 | To keep a speaker entirely in train *or* test | Random split only (leaks speakers) |
| Common 7-label set | `EMOTIONS` list | Four corpora use different label sets | Training one model per corpus |
| Class weights | Give rare classes more influence | *Surprise* is rare | Throwing data away to balance |

## Diagram: from folders to one table

```
 ravdess/…/Actor_01/03-01-06-01-02-01-12.wav ─┐
 cremad/AudioWAV/1001_DFA_ANG_XX.wav          ├─ parse the file name ─▶ one DataFrame `df`
 tess/…/OAF_back_angry.wav                    │     path | emotion | speaker | dataset | label
 savee/ALL/DC_a01.wav                        ─┘
```

## Line-by-line walkthrough

### Imports
```python
import os, sys, json, math, random, time, glob, re, copy, itertools, shutil, warnings
```
Standard-library tools: `os`/`glob` for files and folders, `json` to save results, `math` for π in the spectrogram, `random` for seeds, `time` for timing, `re` for pattern matching file names, `itertools` for combinations (ensemble), `shutil` for copying/zipping.

```python
import numpy as np
import pandas as pd
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
```
`numpy` = numerical arrays (audio is an array), `pandas` = tables (our `df`), `torch` = the deep-learning library, `torchaudio` = audio helpers (mel filters, SpecAugment), `nn` = layers, `F` = functions such as padding and convolution.

```python
import librosa, soundfile as sf
```
`librosa` loads/resamples audio and trims silence; `soundfile` reads/writes WAV files.

```python
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
```
scikit-learn provides the **splitting** tools (random split, speaker-aware folds), the **metrics**, and the **class-weight** formula.

```python
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
```
Hide noisy library warnings; give all figures one clean style.

### Configuration
```python
QUICK_RUN   = False      # True -> tiny subset & 1 epoch: checks the whole pipeline in ~15 min
```
A safety switch: set it to `True` first to verify nothing crashes before spending hours of GPU time.

```python
SEED        = 42
```
The random seed. Fixing it makes splits and initial weights repeatable.

```python
SR          = 16_000                     # target sample rate (Hz) - what all pretrained speech models expect
DUR         = 3.0                        # fixed analysis window (s)
N_SAMPLES   = int(SR * DUR)              # 48 000 samples
```
`SR` is the sample rate. `DUR` is the window length. `N_SAMPLES = 16 000 × 3 = 48 000` numbers per clip – **every model input has exactly this length**.
*Why 3 s?* Most dataset clips are 2–4 s; a fixed length makes batching simple and gives constant latency in the app.

```python
TOP_DB      = 30                         # silence threshold (dB below peak) for librosa.effects.trim
```
Anything quieter than 30 dB below the loudest part at the start or end is treated as silence and removed.

```python
EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID    = {e: i for i, e in enumerate(EMOTIONS)}
N_CLASSES   = len(EMOTIONS)
```
The **7 classes**. `LABEL2ID` turns a name into a number (`"angry" → 0`, …, `"surprise" → 6`) because networks output numbers. The order here is the same order the API returns probabilities.

```python
BATCH_A, LR_A = 64, 2e-3
P1_EPOCHS_A, P2_EPOCHS_A = 30, 20
```
Model A (small CNN): batch of 64 clips, learning rate 0.002, 30 epochs for Protocol 1, 20 for Protocol 2. A small network trains quickly and tolerates a larger learning rate.

```python
HF_IDS = {                                # name -> Hugging Face model id
    "Wav2Vec2-base":   "facebook/wav2vec2-base",
    "HuBERT-base":     "facebook/hubert-base-ls960",
    "WavLM-base-plus": "microsoft/wavlm-base-plus",
}
```
The three pre-trained transformers we compare, by their Hugging Face names (downloaded automatically; this is why Kaggle's *Internet* switch must be on).

```python
BATCH_B, LR_B_BACKBONE, LR_B_HEAD = 16, 5e-5, 1e-3
P1_EPOCHS_B, P2_EPOCHS_B = 12, 6          # P2 uses fewer epochs because it is repeated for 5 folds
```
Model B: smaller batch (16) because the transformer needs more GPU memory; **a tiny learning rate (0.00005) for the pre-trained body** so we gently adapt it without destroying what it learned, and a larger rate (0.001) for the new, randomly initialised head. Protocol 2 repeats training 5 times, so each run uses fewer epochs to fit the time budget.

```python
N_FOLDS     = 5
P2_TRANSFORMERS = 1                       # how many of the best transformers (by P1 *validation* acc) also get Protocol 2
ENABLE_ENSEMBLE = False                   # optional: average transformer probabilities (weights/subset chosen on validation)
NUM_WORKERS = min(4, os.cpu_count() or 2) # CPU workers for on-the-fly augmentation
```
5 folds for speaker-independent testing; only the best transformer (chosen on **validation** accuracy, never the test set) gets this expensive protocol; the ensemble is optional and off; the number of CPU workers that augment audio while the GPU trains (Kaggle has 4 CPU cores).

```python
if QUICK_RUN:
    P1_EPOCHS_A = P2_EPOCHS_A = P1_EPOCHS_B = P2_EPOCHS_B = 1
```
In quick mode every training run lasts a single epoch.

```python
IN_KAGGLE = os.path.isdir("/kaggle/input")
IN_COLAB  = "google.colab" in sys.modules
WORK_DIR  = "/kaggle/working/ser_report" if IN_KAGGLE else ("/content/ser_report" if IN_COLAB else "./ser_report")
```
Detect where the notebook runs and pick a folder for outputs. On Kaggle, `/kaggle/working` is the folder whose content you can download afterwards.

```python
FIG_DIR   = os.path.join(WORK_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TAG = "quick" if QUICK_RUN else "full"
RESULTS_PATH = os.path.join(WORK_DIR, f"results_{TAG}.json")
```
Folder for figures, a tag so quick-run and full-run files never mix, and the path of the results file.

```python
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Seed all random generators (Python, NumPy, PyTorch CPU and GPU) so a rerun gives the same numbers; choose the GPU when present.

```python
RESULTS = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {"p1": {}, "p2": {}}
def save_results(): json.dump(RESULTS, open(RESULTS_PATH, "w"))
```
The **resumable results store**: if the file exists we reload what was finished; `save_results()` is called after every model or fold. If Kaggle disconnects, a rerun skips finished work.

```python
def show(name):
    """Save the current matplotlib figure as a report-ready PNG (200 dpi) and display it."""
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200, bbox_inches="tight"); plt.show()
```
Every figure is saved as a high-resolution PNG for the report *and* shown in the notebook.

### Finding the datasets
```python
def find_root(slug):
    cands = [f"/kaggle/input/{slug}", f"./{slug}"] + glob.glob(f"/kaggle/input/datasets/*/{slug}") + glob.glob(f"/kaggle/input/*/{slug}")
    return next((c for c in cands if os.path.isdir(c)), None)
```
Build a list of places where a dataset folder might be and return the first that exists (or `None`). Kaggle has used several mount layouts over time, so we try them all.

```python
def locate_all():
    return dict(Ravdess=find_root("ravdess-emotional-speech-audio"), Crema=find_root("cremad"),
                Savee=find_root("surrey-audiovisual-expressed-emotion-savee"), Tess=find_root("toronto-emotional-speech-set-tess"))
```
`locate_all` calls `find_root` once per dataset and returns a small dictionary `{"Ravdess": path-or-None, ...}`. The folder names are the Kaggle dataset names (`slug`s). On Colab/local machines, if any entry is `None` the notebook downloads the data with the Kaggle API (`opendatasets`) and calls `locate_all()` again.

```python
assert all(roots.values()), f"Missing dataset(s): {[k for k, v in roots.items() if not v]} - use 'Add Input' on Kaggle"
```
Stop immediately with a clear message if any of the four datasets is missing, instead of failing later with a confusing error.

```python
_rav = glob.glob(os.path.join(roots["Ravdess"], "**", "audio_speech_actors_01-24"), recursive=True)
```
The RAVDESS download also contains **song** recordings. We search for the folder `audio_speech_actors_01-24` and use only speech.

```python
def _wavs(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(".wav"))
```
List every `.wav` file below a folder, in sorted (deterministic) order.

### Label maps
```python
RAV_MAP   = {1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"}   # 2 = calm -> neutral
```
RAVDESS stores the emotion as a number. **Code 2 means "calm"; we map it to "neutral"** (the other datasets have no calm class). This is the deliberate decision the brief required to be documented.

```python
CREMA_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
SAVEE_MAP = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}
```
CREMA-D uses three-letter codes (and has no surprise); SAVEE uses one or two letters.

```python
TESS_MAP  = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
             "sad": "sad", "ps": "surprise", "surprise": "surprise"}
```
TESS writes the emotion as a word; `ps` = "pleasant surprise" becomes *surprise*.

### Reading each file name
```python
rows = []   # (path, emotion, speaker, dataset, was_calm)
for p in _wavs(Ravdess):
    f = os.path.basename(p)[:-4].split("-")
    if len(f) == 7 and int(f[2]) in RAV_MAP:
        rows.append((p, RAV_MAP[int(f[2])], f"RAVDESS_{f[6]}", "RAVDESS", int(f[2]) == 2))
```
`os.path.basename(p)[:-4]` removes the folder and the `.wav` ending, `.split("-")` cuts `03-01-06-01-02-01-12` into 7 fields. Field 3 (`f[2]`) is the emotion code; field 7 (`f[6]`) is the **actor number = speaker**. The last value remembers whether the clip was "calm", only to print how many were merged.

```python
for p in _wavs(Crema):
    f = os.path.basename(p)[:-4].split("_")
    if len(f) >= 3 and f[2] in CREMA_MAP:
        rows.append((p, CREMA_MAP[f[2]], f"CREMAD_{f[0]}", "CREMA-D", False))
```
`1001_DFA_ANG_XX` → speaker `1001`, emotion `ANG`.

```python
for p in _wavs(Tess):
    f = os.path.basename(p)[:-4].split("_")
    if len(f) >= 3 and f[-1].lower() in TESS_MAP:
        rows.append((p, TESS_MAP[f[-1].lower()], f"TESS_{f[0].upper()}", "TESS", False))
```
`OAF_back_angry` → speaker `OAF` (older female) and emotion = last token.

```python
for p in _wavs(Savee):
    f = os.path.basename(p)[:-4].split("_")
    m = re.match(r"[a-z]+", f[1]) if len(f) > 1 else None
    if m and m.group() in SAVEE_MAP:
        rows.append((p, SAVEE_MAP[m.group()], f"SAVEE_{f[0].upper()}", "SAVEE", False))
```
`DC_a01` → speaker `DC`; the regular expression `[a-z]+` takes the letters before the number (`a`, `sa`, …) as the emotion code.

### Building the table
```python
df = pd.DataFrame(rows, columns=["path", "emotion", "speaker", "dataset", "was_calm"])
df["label"] = df.emotion.map(LABEL2ID)
assert df.dataset.nunique() == 4, "all four datasets must be present"
```
Turn the list into a table and add the numeric label. The `assert` guarantees the requirement "never fewer than 4 datasets".

```python
if QUICK_RUN:   # small stratified subset for the smoke test
    df = df.sample(frac=1, random_state=SEED).groupby("emotion").head(90)
df = df.reset_index(drop=True)
```
In quick mode shuffle, then keep at most 90 clips per emotion; finally renumber rows 0…N-1 (many later arrays are indexed by these positions).

### Looking at the data (EDA)
```python
print(pd.crosstab(df.dataset, df.emotion, margins=True))
```
A table counting clips per dataset and emotion – shows imbalance (e.g. CREMA-D has zero *surprise*).

```python
cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=df.label.values)
print("Inverse-frequency class weights:", dict(zip(EMOTIONS, cw.round(2))))
```
The formula `w_c = N / (K · n_c)`: a class with fewer clips gets a larger weight. Here it is only printed for inspection; during training the weights are recomputed from the **training fold only** (so test information never leaks).

## Things to tell the mentor
* **Speaker IDs were extracted on purpose** – they are the foundation of the speaker-independent evaluation.
* **calm → neutral** is an explicit, documented decision.
* **`assert` statements** document assumptions and fail early.
* The results file makes the notebook **resumable** on a free GPU platform.
