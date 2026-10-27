# 1. The machine-learning part

Notebook: [`ser_report_notebook.ipynb`](../../ser_report_notebook.ipynb) (generated from `tools/report_notebook.py`).
It runs on a free **Kaggle GPU (T4)** and produces: the trained models, all evaluation numbers, the report figures and the `model.onnx` file used by the cloud API.

## Purpose

Turn 3-second audio clips into one of **seven emotions** (angry, disgust, fear, happy, neutral, sad, surprise) and measure honestly how well that works on **new people**.

## The workflow in one picture

```
 §2 config        §3 datasets            §4 preprocess         §5 augment (train only)
┌──────────┐    ┌───────────────┐      ┌─────────────────┐    ┌─────────────────────┐
│ constants│──▶│ find 4 corpora │────▶│ 16 kHz, trim,   │───▶│ gain / noise /      │
│ seeds    │   │ parse labels + │      │ 3 s window,     │    │ pitch / time-stretch│
│ paths    │   │ speaker IDs    │      │ cache as int16  │    │ on the fly          │
└──────────┘   └───────────────┘      └─────────────────┘    └──────────┬──────────┘
                                                                         ▼
 §13 export          §8-11 evaluate                           §6-7 models and training
┌──────────────┐   ┌────────────────────────────┐          ┌──────────────────────────┐
│ ONNX + int8  │◀─│ Protocol 1: random split    │◀────────│ A: CNN + BiLSTM + Attn    │
│ VAD demo     │   │ Protocol 2: unseen speakers │         │ B: Wav2Vec2 / HuBERT /    │
│ zip bundle   │   │ per-dataset + error analysis│         │    WavLM (fine-tuned)     │
└──────────────┘   └────────────────────────────┘          └──────────────────────────┘
```

## Map: notebook section → documentation page

| Notebook § | Content | Page |
|---|---|---|
| 1 | Introduction, objectives, design decisions | (markdown only; summarised in [../README.md](../README.md)) |
| 2 | Environment and configuration | [01-data-and-labels.md](01-data-and-labels.md) |
| 3 | Datasets and label harmonisation | [01-data-and-labels.md](01-data-and-labels.md) |
| 4–5 | Pre-processing and augmentation | [02-preprocessing-augmentation.md](02-preprocessing-augmentation.md) |
| 6 | Model architectures | [03-models.md](03-models.md) |
| 7–11 | Training, evaluation protocols, analysis | [04-training-evaluation.md](04-training-evaluation.md) |
| 12 | Auto-generated final results and abstract | [04-training-evaluation.md](04-training-evaluation.md) |
| 13 | Export, quantisation, real-time demo | [05-export-and-realtime.md](05-export-and-realtime.md) |
| 14 | Conclusion, limitations, references | (markdown only) |

## Why this design (components at a glance)

| Component | What | Why |
|---|---|---|
| `QUICK_RUN` switch | Runs the whole pipeline on ~600 clips and 1 epoch | Catches bugs in 15 minutes instead of after 6 hours of GPU time |
| Results saved to JSON after every model/fold | Resumable | Kaggle sessions can stop; nothing is lost |
| Cached waveforms as `int16` `.npy` | Decode audio once, memory-map afterwards | Loading 12 000 WAV files every epoch would be far too slow |
| One generic `train_model` for all models | Same optimiser, loss, schedule | Differences in results come from the model, not from tuning tricks |
| Validation set for model selection, test set used once | Prevents optimistic bias | A test set used for choices is no longer a test set |
| Figures saved as PNG (`show()`) | Ready for the project report | No re-plotting needed |

## How to explain it to the mentor in 60 seconds

> "We combined four emotional-speech datasets into one seven-class problem. Audio is cleaned and cut to 3 seconds, and we augment only the training data to avoid leakage. We trained a small CNN-BiLSTM-attention network and three large pre-trained speech transformers, and evaluated them twice: once with a random split, which is the usual headline number, and once with speaker-independent cross-validation, which tells us how the system behaves for a new user. The best speaker-independent model was exported to ONNX and deployed."
