# 1b. Pre-processing and augmentation (notebook §4–5)

## Purpose
1. Turn every audio file into **exactly the same format**: 16 kHz, mono, speech only, 3 seconds long.
2. Cache the result so training is fast.
3. Create extra, slightly modified training examples (**augmentation**) — *only for the training data*.

## Why this component

| Piece | What | Why | Rejected alternative |
|---|---|---|---|
| Resample to 16 kHz mono | Same sample rate for all files | The pretrained transformers were trained on 16 kHz; datasets differ (22 kHz, 44 kHz …) | Feeding mixed rates: wrong results |
| Trim silence | Remove quiet start/end | Silence carries no emotion and wastes the 3 s window | Keeping silence: model learns recording-room noise |
| Fixed 3 s window | Crop or pad | Batching needs equal lengths; constant latency in the app | Variable lengths: complex batching, variable latency |
| Cache as `int16` `.npy` | Decode once | 12 000 files decoded every epoch would dominate run time | Re-reading WAVs each epoch |
| Augmentation in `Dataset` | Random changes on the fly | More variety → less over-fitting; robust to microphones/noise | Pre-augmenting before splitting = **data leakage** |
| `train`-only augmentation | Val/test stay untouched | Evaluation must reflect real, unmodified audio | Augmenting everything |

## Diagram: the one function used in training *and* in the API

```
 raw audio file
      │ librosa.load(sr=16000, mono=True)       → resample + mix to mono
      ▼
 float array of any length
      │ librosa.effects.trim(top_db=30)         → cut silent edges
      ▼
 speech-only array
      │ centre-crop if > 3 s, zero-pad if < 3 s
      ▼
 exactly 48 000 samples  ──▶ model
```
The backend re-implements the same steps (see [../02-backend/README.md](../02-backend/README.md)) so that the app's audio is treated exactly like the training audio. This is called **train/serve parity**.

## Line-by-line walkthrough

### `to_fixed`
```python
def to_fixed(y, n=N_SAMPLES):
```
`y` is the audio as a NumPy array; `n` is the target length (48 000).

```python
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
```
Safety net: a broken or empty file becomes a silent 3 s clip instead of crashing.

```python
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
```
Removes leading and trailing parts that are more than 30 dB quieter than the loudest part. It returns the trimmed audio and the index range (ignored, hence `_`).

```python
    if len(yt) < 0.1 * SR:                 # trimming removed (almost) everything -> keep the original
        yt = y
```
If trimming left less than 0.1 s (a very quiet or whispery clip) we keep the original rather than feeding an almost empty signal.

```python
    if len(yt) >= n:                       # centre crop
        s = (len(yt) - n) // 2
        yt = yt[s:s + n]
```
Longer than 3 s: take the **middle** 3 s (`s` = how much to skip at the start). The middle usually contains the strongest emotional content.

```python
    else:                                  # zero-pad at the end
        yt = np.pad(yt, (0, n - len(yt)))
    return yt.astype(np.float32)
```
Shorter than 3 s: add zeros at the end. Return 32-bit floats because that is what the network expects.

### `load_clip` and the cache
```python
def load_clip(path):
    y, _ = librosa.load(path, sr=SR, mono=True)            # resample to 16 kHz mono
    return (np.clip(to_fixed(y), -1, 1) * 32767).astype(np.int16)    # int16 cache saves 50 % memory
```
Load + resample + mix to mono in one call; run `to_fixed`; clip values into [−1, 1]; multiply by 32767 and store as 16-bit integers (half the memory of 32-bit floats, no audible loss).

```python
WAV_PATH, META_PATH = os.path.join(WORK_DIR, f"waves_{TAG}.npy"), os.path.join(WORK_DIR, f"meta_{TAG}.csv")
if os.path.exists(WAV_PATH) and os.path.exists(META_PATH) and len(pd.read_csv(META_PATH)) == len(df):
    print("loading cached waveforms")
```
If a cache from an earlier run exists *and* has the same number of rows as the current table, reuse it.

```python
    waves = Parallel(n_jobs=-1, batch_size=32)(delayed(load_clip)(p) for p in df.path)
    np.save(WAV_PATH, np.stack(waves)); df.to_csv(META_PATH, index=False)
```
Otherwise process all files **in parallel on every CPU core** (`joblib`), stack them into one `(N, 48000)` array and save it together with the metadata table.

```python
WAVES = np.load(WAV_PATH, mmap_mode="r")            # (N, 48000) int16, memory-mapped
LABELS, SPEAKERS, DATASETS = df.label.values, df.speaker.values, df.dataset.values
```
`mmap_mode="r"` means the array stays on disk and is read page by page, so RAM stays small. We also keep the labels, speakers and dataset names as plain arrays; **all later splitting works with row numbers into these arrays**.

### The mel spectrogram figure
```python
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=64), ref=np.max)
```
A *mel spectrogram* is a picture: x = time, y = pitch (on the ear-like mel scale), brightness = energy. `n_fft=1024` is the analysis window (64 ms), `hop_length=256` moves it 16 ms each step, `n_mels=64` frequency bands; `power_to_db` converts to decibels (log scale), which matches how we hear loudness. This figure is for **understanding**; the model computes its own version inside the network.

## Augmentation

### `augment`
```python
def augment(y, rng):
```
`rng` is a random-number generator passed in so each call is independent.

```python
    if rng.random() < 0.5:                                   # random gain
        y = y * rng.uniform(0.5, 1.5)
```
With 50 % probability multiply the volume by a random factor between 0.5× and 1.5× (microphones record at different levels).

```python
    if rng.random() < 0.5:                                   # white Gaussian noise at a random SNR
        snr_db = rng.uniform(15, 35)
        y = y + rng.normal(0, 1, len(y)) * np.sqrt((np.mean(y ** 2) + 1e-10) / 10 ** (snr_db / 10))
```
With 50 % probability add random noise. The noise level is set by a *signal-to-noise ratio* between 15 and 35 dB: noise power = signal power / 10^(SNR/10). The `1e-10` prevents division problems for silent input.

```python
    if rng.random() < 0.3:                                   # pitch shift +-2 semitones
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=rng.uniform(-2, 2))
```
30 %: raise or lower the pitch by up to two semitones without changing speed (simulates different speakers).

```python
    if rng.random() < 0.3:                                   # time stretch 0.9-1.1x
        y = librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))
```
30 %: speak 10 % faster or slower without changing pitch.

```python
    return _fix_len(np.clip(y, -1, 1)).astype(np.float32)
```
Time stretching changes the length, so `_fix_len` crops/pads back to 48 000 samples; clipping keeps values in [−1, 1].

*Why these probabilities?* Pitch shift and time stretch are the most "aggressive" (and slowest) changes, so they are applied less often (30 %) than gain and noise (50 %).

### `WaveDataset` and `make_loader`
```python
class WaveDataset(torch.utils.data.Dataset):
```
PyTorch's standard way to feed data: a class that knows its length and can return item number *k*.

```python
    def __init__(self, idx, augment_on=False):
        self.idx, self.augment_on = np.asarray(idx), augment_on
```
`idx` = the row numbers this dataset is allowed to use (e.g. only training rows). `augment_on` says whether to augment.

```python
    def __len__(self): return len(self.idx)
```
Tells the `DataLoader` how many items exist (needed to know how many batches make an epoch).

```python
    def __getitem__(self, k):
        j = self.idx[k]
        y = WAVES[j].astype(np.float32) / 32768.0
```
Look up the real row `j`, read its int16 samples and convert back to floats in [−1, 1].

```python
        if self.augment_on:
            y = augment(y, np.random.default_rng())          # fresh randomness per call (safe with workers)
        return torch.from_numpy(y), int(LABELS[j])
```
Augment only for training; every call uses a **new** generator so parallel worker processes don't produce identical "random" changes. Return the audio tensor and its label.

```python
def make_loader(idx, batch_size, train):
    return torch.utils.data.DataLoader(WaveDataset(idx, augment_on=train), batch_size=batch_size, shuffle=train,
                                       num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                                       persistent_workers=NUM_WORKERS > 0)
```
A `DataLoader` groups items into batches. `shuffle=train` reshuffles only the training data each epoch; `num_workers` augments in background processes while the GPU is busy; `pin_memory` speeds up CPU→GPU copies; `persistent_workers` keeps the workers alive between epochs. **`augment_on=train`** is the line that guarantees *validation and test data are never augmented*.

## Things to tell the mentor
* **Why augmenting after the split matters:** it is the single biggest data-leakage trap in tutorials; the notebook avoids it by design.
* **Train/serve parity:** `to_fixed` is shared with the API, so the app is judged by the same preprocessing as the experiments.
* **int16 + memory-mapping** is an engineering choice that makes a 12 k-clip dataset fit comfortably into free-tier RAM.
