# 1e. Export, quantisation and real-time inference (notebook §13)

## Purpose
Turn the best trained PyTorch model into a **single portable file** (`model.onnx`, optionally `model_int8.onnx`) that a tiny CPU server can run without PyTorch, check that **nothing was lost** in the conversion, and demonstrate **real-time** use with voice-activity detection.

## Why this component

| Piece | What | Why | Rejected alternative |
|---|---|---|---|
| Export from the best **Protocol-2** fold | The deployed model is the best speaker-independent model | It is the one that generalises to new people | Deploying the best random-split model |
| ONNX format | Portable graph of the network | Runs in ONNX Runtime without PyTorch | Shipping PyTorch (hundreds of MB) |
| `attn="eager"` for export | Plain attention implementation | The optimised implementation does not export reliably | Default attention: export errors |
| Dynamic batch axis | `(batch, 48000)` with variable batch | Server can send 1 or several windows | Fixed batch of 1 |
| int8 quantisation of MatMul only | Weights stored as 8-bit | ≈ 4× smaller, faster on CPU | Quantising convolutions too: hurts accuracy |
| Fidelity table (PyTorch vs ONNX vs int8) | Measures accuracy, agreement, size, latency on real unseen-speaker clips | Never assume a conversion is lossless | Trusting the converter |
| Fallback to the next candidate | If an export fails, try the next best model | A pipeline that survives one failure | Crashing after 6 hours |
| silero-VAD + sliding window | Speech detection + 3 s windows with 50 % overlap | Real-time use on streams; skip silence | Classifying silence |
| Zip bundle | Figures, results, model files in one download | Easy hand-over for the report/deployment | Collecting files one by one |

## Diagram

```
best_p2_<model>.pt  (PyTorch weights of the best unseen-speaker fold)
        │ build_for_export  →  same architecture, attention="eager"
        ▼
 torch.onnx.export  ──▶  model.onnx  (float32, ≈ 4 bytes/weight)
        │ quantize_dynamic(MatMul only, int8, per-channel)
        ▼
 model_int8.onnx (≈ 1/4 size)
        │
        ├─ fidelity check on up to 300 real unseen-speaker clips:
        │     accuracy, agreement with PyTorch, size, 1-thread latency
        ▼
 recommended file = int8 if accuracy loss < 1 point, else float32
        ▼
 labels.json (class order, sample rate, window length, which file to serve)
```

## Line-by-line walkthrough

### Choosing and building the model
```python
exp_candidates = sorted(p2_models, key=lambda a: -RESULTS["p2"][a]["mean"])
```
List all models that finished Protocol 2, **best speaker-independent mean first**.

```python
def build_for_export(arch):
    return CNNBiLSTMAttention() if ARCH[arch]["kind"] == "cnn" else SSLClassifier(arch, attn="eager")
```
Create an empty copy of the same architecture. The CNN is the same class; transformers are built with `attn="eager"` (the simple attention maths that ONNX understands). The `kind` field from the `ARCH` registry decides which.

### The export loop
```python
EXPORTED = None
for arch in exp_candidates:                       # fall back to the next candidate if an export fails
    try:
        m = build_for_export(arch); m.load_state_dict(torch.load(os.path.join(WORK_DIR, f"best_p2_{arch}_{TAG}.pt"), map_location="cpu")); m.eval().cpu()
```
For each candidate: build the model, **load the saved best-fold weights**, switch to evaluation mode (no dropout/masking) and move to the CPU (export is traced on CPU).

```python
        kw = dict(input_names=["waveform"], output_names=["logits"], dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
```
Names for the input and output (the API uses `"waveform"`), a **dynamic batch dimension** (axis 0 may be any size), and **opset 17** – the version of the ONNX operator set (stable and supported by ONNX Runtime).

```python
        try:    torch.onnx.export(m, (torch.randn(1, N_SAMPLES),), ONNX_PATH, dynamo=False, **kw)
        except TypeError: torch.onnx.export(m, (torch.randn(1, N_SAMPLES),), ONNX_PATH, **kw)
```
`torch.onnx.export` **traces** the model by running a dummy 3-second input through it and records every operation. `dynamo=False` selects the classic, reliable exporter (newer PyTorch versions have a new one that handles transformers less reliably); older PyTorch versions do not know that argument, hence the `TypeError` fallback.

```python
        EXPORTED = arch; EXPORT_MODEL = m; print("exported:", arch, f"({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)"); break
    except Exception as e:
        print(f"export of {arch} failed ({repr(e)[:160]}) -> trying next candidate")
assert EXPORTED, "no model could be exported"
```
On success remember which model was exported and stop; on failure print a short error and try the next candidate. If none works the notebook stops with a clear message.

### Quantisation
```python
try:
    quantize_dynamic(ONNX_PATH, INT8_PATH, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=True)
except Exception as e:
    print("INT8 quantisation failed:", repr(e)[:200]); INT8_PATH = None
```
**Dynamic quantisation** stores weights as `int8` (a number between −128 and 127 plus a scale factor) and converts activations on the fly. Settings: only `MatMul` operations are quantised (these are the large linear layers of the transformer, where almost all weights are); `per_channel=True` uses one scale per output channel, which is more accurate than one scale for the whole matrix. If quantisation fails the notebook continues with the float32 file.

### Fidelity check: is the exported model still the same model?
```python
best_fold = RESULTS["p2"][EXPORTED]["best_fold"]
fold_rec = next(f for f in RESULTS["p2"][EXPORTED]["folds"] if f["fold"] == best_fold)
chk_idx = np.array(fold_rec["idx"])[:300]
```
Take the **test clips of the best fold** — speakers the exported model never saw — up to 300 of them.

```python
def run_onnx(path, x, bs=8, threads=2):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    out = np.concatenate([s.run(None, {"waveform": x[i:i + bs]})[0] for i in range(0, len(x), bs)])
```
Create an ONNX Runtime session on the CPU and score the clips in batches of 8. `s.run(None, {"waveform": …})` is **exactly the call the API makes**.

```python
    s1 = ort.SessionOptions(); s1.intra_op_num_threads = 1; s1 = ort.InferenceSession(path, s1, providers=["CPUExecutionProvider"])
    s1.run(None, {"waveform": x[:1]}); t0 = time.time(); [s1.run(None, {"waveform": x[:1]}) for _ in range(3)]
    return out, (time.time() - t0) / 3 * 1000
```
Measure **latency**: a second session with **one thread** (a stand-in for a small free server), one warm-up run (the first call is always slower), then the average of 3 timed runs in milliseconds.

```python
with torch.no_grad(): ref_logits = np.concatenate([EXPORT_MODEL(torch.from_numpy(xb[i:i + 8])).numpy() for i in range(0, len(xb), 8)])
```
The **reference**: the original PyTorch model's outputs on the same clips.

```python
    fid.append({"Variant": name, "Accuracy (%)": round((out.argmax(1) == yb).mean() * 100, 2), "Agreement with PyTorch (%)": round((out.argmax(1) == ref_logits.argmax(1)).mean() * 100, 2),
                "Size (MB)": round(os.path.getsize(path) / 1e6, 1), "Latency 1 thread (ms)": round(lat)})
```
For each exported file: accuracy against the true labels, **agreement** with the PyTorch prediction (should be ≈ 100 % for float32), file size and latency. The table lets you state the quantisation trade-off with numbers.

### The recommendation rule and `labels.json`
```python
use_int8 = INT8_PATH is not None and "ONNX int8 (MatMul only)" in fid_df.index and \
           fid_df.loc["ONNX fp32", "Accuracy (%)"] - fid_df.loc["ONNX int8 (MatMul only)", "Accuracy (%)"] < 1.0
SERVE_PATH = INT8_PATH if use_int8 else ONNX_PATH
```
**Use int8 only if it loses less than one accuracy point**; otherwise stay with float32. The decision is automatic and based on measured data.

```python
json.dump({"labels": EMOTIONS, "sample_rate": SR, "window_samples": N_SAMPLES, "top_db": TOP_DB, "architecture": EXPORTED,
           "protocol2_mean_acc": RESULTS["p2"][EXPORTED]["mean"], "protocol1_acc": RESULTS["p1"].get(EXPORTED, {}).get("acc"),
           "recommended_file": os.path.basename(SERVE_PATH)}, open(os.path.join(EXPORT_DIR, "labels.json"), "w"), indent=2)
```
`labels.json` is the **contract between the notebook and the API**: class order (so output index 3 means "happy"), sample rate, window length, trimming threshold, which architecture it is, and which file to serve. The backend reads it at start-up (`meta = json.load(...)`), so the API never hard-codes these values.

## Real-time inference with VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps
VAD = load_silero_vad()
SESS = ort.InferenceSession(SERVE_PATH, providers=["CPUExecutionProvider"])
```
**silero-VAD** is a small pre-trained neural network that finds *where someone is speaking*. `SESS` is the ONNX session of the model we are going to deploy — so the demo uses the real deployment file.

```python
def classify_window(y):
    logits = SESS.run(None, {"waveform": to_fixed(y)[None]})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return EMOTIONS[int(p.argmax())], float(p.max()), p
```
`to_fixed(y)[None]` prepares one window with the shared preprocessing and adds the batch dimension. The two `p` lines are a hand-written **softmax** (subtracting the maximum first prevents overflow). Returns the emotion name, its probability (**confidence**) and all probabilities.

```python
def stream_emotions(y, win_s=3.0, hop_s=1.5, min_speech=0.3, verbose=True):
    ts = get_speech_timestamps(torch.from_numpy(y), VAD, sampling_rate=SR)
    speech = np.zeros(len(y), dtype=bool)
    for t in ts: speech[t["start"]:t["end"]] = True
```
Ask the VAD for speech segments (start/end sample numbers) and mark those samples `True` in a yes/no array the same length as the audio.

```python
    W, H, out = int(win_s * SR), int(hop_s * SR), []
    for s in range(0, max(1, len(y) - W // 2), H):
        seg, sp = y[s:s + W], speech[s:s + W]
```
`W` = window length in samples (3 s), `H` = hop (1.5 s = **50 % overlap**). Slide the window along the signal; `seg` is the audio, `sp` the speech marks inside it.

```python
        if rec["speech_frac"] >= min_speech:
            t0 = time.time(); e, c, _ = classify_window(seg); rec.update(emotion=e, confidence=c, latency_ms=(time.time() - t0) * 1000)
```
Only windows in which at least **30 % is speech** are classified; silent windows are skipped (they would produce meaningless predictions) and the latency of each real prediction is recorded.

*Why 50 % overlap?* Emotion can change; overlapping windows give a new reading every 1.5 s and avoid missing a change that falls on a window border.

## Bundle for hand-over
```python
zip_path = shutil.make_archive(os.path.join(os.path.dirname(OUT_DIR), "ser_report_bundle"), "zip", OUT_DIR)
```
Packs the model files, figures, `results_for_report.json` and `report_summary.md` into one `.zip` that you download from Kaggle's *Output* tab.

## Things to tell the mentor
* The **deployed model comes from Protocol 2**, i.e. it is chosen for generalisation to new speakers.
* **We measured the cost of compression** (accuracy, agreement, size, latency) before choosing the served file.
* `labels.json` documents **class order and preprocessing**, preventing a classic bug: a server that preprocesses differently from training.
* The real-time demo uses the **same ONNX file as the cloud API**.
