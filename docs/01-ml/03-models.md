# 1c. The two model families (notebook §6)

## Purpose
Define two kinds of classifier that both take a **3-second waveform `(batch, 48000)`** and return **7 scores (logits)**:

* **Model A – CNN + BiLSTM + Attention** on a mel spectrogram: small, quick, deployable anywhere.
* **Model B – self-supervised transformer** (Wav2Vec2, HuBERT or WavLM) fine-tuned for emotion: large, most accurate.

## Why this component

| Piece | What | Why | Trade-off |
|---|---|---|---|
| Model A baseline | ≈ 0.95 M trainable parameters (+1.1 M fixed spectrogram constants), 8 MB as ONNX | Fits any free server; shows what a classical design achieves | Lower accuracy than transformers |
| Model B transformers | ~95 M parameters, pre-trained on unlabeled speech | Already "understands" speech → small labelled data is enough | Big: needs int8 + a small server budget |
| Mel front-end *inside* the model | Spectrogram computed with convolutions | The exported ONNX file takes raw audio, so the API needs no spectrogram code | Slightly unusual implementation |
| Same input/output for both | `(B, 48000) → (B, 7)` | One training loop, one export function, one API | – |
| Weighted sum of layers | The model learns which transformer layers to use | Emotion lives in lower/middle layers, not only the last | Few extra parameters |
| Three backbones | Wav2Vec2, HuBERT, WavLM | Compare pre-training methods fairly | 3× the training time |
| Frozen CNN feature encoder | Early audio convolutions not trained | Standard practice: stable, saves memory | Slightly less adaptation |

## Diagram

```
Model A                                              Model B
waveform (B,48000)                                   waveform (B,48000)
   │ utt_norm                                           │ utt_norm
   ▼                                                    ▼
MelFrontEnd (STFT→mel→log)                         pre-trained transformer
   │  (B,1,64,frames)                               (CNN encoder frozen + 12 layers fine-tuned)
   ▼                                                    │ 13 hidden states
4 × [Conv→BatchNorm→ReLU→MaxPool→Dropout]              ▼
   ▼                                              weighted sum of the 13 (learned)
BiLSTM (reads the sequence both ways)                   ▼
   ▼                                              mean over time → LayerNorm
Attention pooling (learn where to look)                 ▼
   ▼                                              Linear 768→256 → ReLU → Dropout → Linear 256→7
Dropout → Linear 256→64 → ReLU → Dropout → Linear 64→7
   ▼                                                    ▼
7 logits                                            7 logits
```

## Line-by-line walkthrough

### `draw_pipeline` — the system diagram figure
```python
def draw_pipeline():
    """Figure: end-to-end system diagram drawn with matplotlib (no external assets)."""
```
A helper that **draws the end-to-end system diagram** (microphone → upload → trim → transformer → softmax → average) with Matplotlib rectangles and arrows and saves it as `fig04_system_pipeline.png` via `show()`. It needs no image files, so the notebook stays self-contained; the figure is meant for the report.

### `utt_norm` — shared by both models
```python
def utt_norm(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-7)
```
For every clip: subtract its average (remove DC offset), then divide by its RMS (root-mean-square loudness). Result: zero mean and unit loudness. **Why:** different microphones record at different levels; the model should respond to *how* someone speaks, not how loud the phone is. The `1e-7` avoids dividing by zero for silence. Because this is *inside* the model, it also runs inside the deployed ONNX file.

### `MelFrontEnd` — a spectrogram built from convolutions
```python
class MelFrontEnd(nn.Module):
    def __init__(self, sr=SR, n_fft=1024, hop=256, n_mels=64, f_min=20., f_max=8000.):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        win = torch.hann_window(n_fft)
```
Settings: 1024-sample analysis window (64 ms), 256-sample step (16 ms), 64 mel bands covering 20 Hz–8 kHz (speech lives below 8 kHz). A **Hann window** smoothly fades each analysis window in and out to avoid artefacts at the edges.

```python
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float(); n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * math.pi * k * n / n_fft
        self.register_buffer("cos_k", (torch.cos(ang) * win).unsqueeze(1))        # (F,1,n_fft)
        self.register_buffer("sin_k", (-torch.sin(ang) * win).unsqueeze(1))
```
The Fourier transform measures "how much of frequency *k* is in this window" by multiplying with cosine and sine waves. We pre-compute those waves (`cos_k`, `sin_k`) as **fixed convolution kernels**. `register_buffer` stores them with the model (saved and exported) but they are **not trainable**.

```python
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sr, norm="slaney", mel_scale="slaney")
        self.register_buffer("mel_fb", fb)                                        # (F, n_mels)
```
The **mel filter-bank**: a matrix that merges the 513 linear frequency bins into 64 ear-like bands.

```python
    def forward(self, x):                                                         # x: (B, T)
        x = F.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
```
Add a channel dimension and mirror-pad the edges so the first and last windows are centred properly.

```python
        re_, im_ = F.conv1d(x, self.cos_k, stride=self.hop), F.conv1d(x, self.sin_k, stride=self.hop)
        power = re_ ** 2 + im_ ** 2                                               # |STFT|^2   (B,F,frames)
```
Convolve with the cosine/sine kernels, stepping 256 samples at a time = the **real and imaginary parts of the STFT**. Squaring and adding gives the **power** (energy) at each frequency and time.

```python
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)    # (B, n_mels, frames)
        lm = torch.log(mel + 1e-6)
```
Apply the mel filter-bank (matrix multiply) and take the logarithm (human loudness perception is logarithmic; `1e-6` avoids `log(0)`).

```python
        lm = (lm - lm.mean(dim=(1, 2), keepdim=True)) / torch.sqrt(lm.var(dim=(1, 2), keepdim=True, unbiased=False) + 1e-5)
        return lm.unsqueeze(1)                                                    # (B, 1, n_mels, frames)
```
Standardise each spectrogram (mean 0, variance 1) and add a channel dimension so it looks like a one-channel image for the CNN.

*Why not `torchaudio.transforms.MelSpectrogram`?* Its FFT operator is hard to export to ONNX. Convolutions and matrix multiplications export cleanly, so the **whole model, including feature extraction, becomes one ONNX file**.

### `CNNBiLSTMAttention` — Model A
```python
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10, iid_masks=True)   # SpecAugment
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20, iid_masks=True)
```
**SpecAugment**: during training randomly hide a stripe of frequencies (up to 10 bands) and a stripe of time (up to 20 frames) in each spectrogram. It forces the model not to rely on any single region and fights over-fitting.

```python
        def block(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(pool), nn.Dropout2d(0.1))
```
One CNN block: **Conv2d** (3×3 filters detect local patterns such as pitch movements) → **BatchNorm** (stabilises training; `bias=False` because BatchNorm adds its own offset) → **ReLU** (non-linearity) → **MaxPool** (keeps the strongest response, halves the size) → **Dropout2d** (randomly drops whole feature maps to reduce over-fitting).

```python
        self.cnn = nn.Sequential(block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 128, (2, 1)), block(128, 128, (2, 1)))
```
Four blocks with more and more filters (32 → 64 → 128 → 128). The first two pool both frequency and time; the last two pool only frequency, so the **time resolution is kept for the LSTM**. After the four blocks the 64 mel bands are reduced by 2⁴ = 16 → 4 bands.

```python
        self.lstm = nn.LSTM(128 * (n_mels // 16), hidden, batch_first=True, bidirectional=True)
```
The **BiLSTM** reads the frame sequence forwards *and* backwards. Input size = 128 channels × 4 bands = 512 values per frame; `hidden=128` per direction → 256 outputs per frame. **Why:** emotion shows in *how the voice changes over time* (rising pitch, speeding up), which a recurrent layer captures.

```python
        self.att = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1))            # additive attention
```
A tiny network that gives each frame a score "how important is this moment".

```python
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
```
The classifier: dropout → 256→64 → ReLU → dropout → 64→7 logits.

```python
    def forward(self, wav):
        x = self.frontend(utt_norm(wav))                         # (B,1,M,T)
        if self.training:                                        # SpecAugment (training only)
            x = self.time_mask(self.freq_mask(x))
        x = self.cnn(x)                                          # (B,C,M/16,T/4)
```
Normalise → spectrogram → (training only) mask → CNN. `self.training` is `True` after `model.train()` and `False` after `model.eval()`, so masking never happens at test or inference time.

```python
        b, c, f, t = x.shape
        h, _ = self.lstm(x.permute(0, 3, 1, 2).reshape(b, t, c * f))   # (B,T',2H)
```
Rearrange from (batch, channels, freq, time) to (batch, time, channels×freq) – the sequence layout the LSTM expects – and run it.

```python
        w = torch.softmax(self.att(h), dim=1)                    # attention weights over time
        return self.head((w * h).sum(dim=1))
```
Score every time step, `softmax` turns scores into weights that add up to 1, multiply each frame by its weight and **sum over time** to one 256-number summary of the whole clip; the head turns that into 7 logits.

```python
    def param_groups(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": LR_A}]
```
Tells the optimiser which parameters to train and with which learning rate (one group here).

### `SSLClassifier` — Model B
```python
from transformers import AutoModel
```
Hugging Face's loader: gives the right class (Wav2Vec2Model, HubertModel, WavLMModel) for a model name.

```python
        kw = dict(layerdrop=0.0, mask_time_prob=0.05, mask_time_length=5)     # light time-masking acts as regularisation
        if attn: kw["attn_implementation"] = attn                              # "eager" is the ONNX-friendly implementation
        self.backbone = AutoModel.from_pretrained(HF_IDS[name], **kw)
```
Download the pre-trained model. `layerdrop=0` keeps all layers active (stable); `mask_time_prob=0.05` randomly hides ~5 % of time steps in training (a mild regulariser built into these models). When exporting we pass `attn="eager"`, the attention implementation that converts to ONNX reliably.

```python
        for prm in self.backbone.feature_extractor.parameters():               # the low-level CNN feature encoder stays frozen
            prm.requires_grad = False                                          # (done by hand: not every backbone class offers freeze_feature_encoder)
```
**Freeze** the convolutional feature encoder (the first stage that turns raw audio into frames). `requires_grad=False` means "do not update". *Why:* these low-level filters are generic; freezing is the standard recipe, saves memory and makes training stable.

```python
        d, n_layers = self.backbone.config.hidden_size, self.backbone.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(n_layers))                     # a_l (softmax -> weights); starts uniform
```
`d = 768` (feature size), `n_layers = 12 + 1 = 13` hidden states (the output of the embedding stage plus each of the 12 transformer layers). `layer_w` is a learnable vector of 13 numbers, initialised to zeros → after `softmax` every layer starts with the same weight 1/13.

```python
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))
```
LayerNorm stabilises the pooled features; the head maps 768 → 256 → 7.

```python
    def forward(self, wav):
        hs = self.backbone(utt_norm(wav), output_hidden_states=True).hidden_states     # tuple of (B,T',d)
```
Run the transformer and ask it to return **all** intermediate layers, not just the last.

```python
        w = torch.softmax(self.layer_w, dim=0).view(-1, 1, 1, 1)
        h = (w * torch.stack(hs, dim=0)).sum(dim=0)                                    # weighted sum of layers
        return self.head(self.norm(h.mean(dim=1)))                                     # mean-pool over time
```
Turn the 13 learnable numbers into weights summing to 1; stack the 13 layers into one tensor, multiply each by its weight and add them (the **learned layer mix**); average over time to a single 768-vector per clip; normalise; classify. *Why mix layers:* research on speech models shows that lower/middle layers keep prosody and voice quality — exactly what emotion needs — while the top layers become more about words.

```python
    def param_groups(self):
        back = [p for p in self.backbone.parameters() if p.requires_grad]
        rest = [self.layer_w] + list(self.norm.parameters()) + list(self.head.parameters())
        return [{"params": back, "lr": LR_B_BACKBONE}, {"params": rest, "lr": LR_B_HEAD}]
```
**Two learning rates:** a tiny one (5e-5) for the pre-trained body, a larger one (1e-3) for the new parts. This is called *discriminative learning rates* and is a standard fine-tuning technique.

### The registry and the summary table
```python
ARCH = {"CNN-BiLSTM-Attn": dict(build=lambda: CNNBiLSTMAttention(), amp=False, bs=BATCH_A, ep1=P1_EPOCHS_A, ep2=P2_EPOCHS_A, kind="cnn")}
for _n in HF_IDS:
    ARCH[_n] = dict(build=(lambda n=_n: SSLClassifier(n)), amp=True, bs=BATCH_B, ep1=P1_EPOCHS_B, ep2=P2_EPOCHS_B, kind="ssl")
```
A dictionary describing every model: how to build it (`build`), whether to use **mixed precision** (`amp`: 16-bit maths on the GPU for speed and memory — used for the transformers), batch size, epochs for each protocol, and the `kind` (used later when exporting). The `lambda n=_n` trick fixes the current loop value; without it all lambdas would build the *last* model.

```python
    m = a["build"](); out = m.eval()(torch.randn(2, N_SAMPLES))
```
Build every model and push random audio through it: this checks that **all models accept `(2, 48000)` and output `(2, 7)`** before spending hours on training. The surrounding lines also print parameter counts (total and trainable).

## Things to tell the mentor
* *Why two models?* The small one proves the idea is feasible on a tiny server; the large one shows what modern pre-trained speech models add.
* *Why not train a transformer from scratch?* It needs tens of thousands of hours of speech; fine-tuning a pre-trained one needs only our ~12 k clips.
* *Why is the mel front-end inside the network?* One file → one export → no feature code in the API → no train/serve mismatch.
