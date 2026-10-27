# EmoSense — Speech Emotion Recognition for Health & Wellness
### Documentation of every component and *why* it exists

This folder explains the whole project: the machine-learning code, the cloud API, the deployment, and the Android app.
Every page follows the same template:

1. **Purpose** – what the component does in one paragraph.
2. **Why this component** – a table: *what it is → why we chose it → what we rejected → trade-off*.
3. **Diagram** – how the parts connect.
4. **Line-by-line walkthrough** – the real code is quoted, followed by *what it does* and *why*.
5. **Things to tell the mentor** – the points that show you understand the design.

## Read in this order

| # | Page | What you will learn |
|---|---|---|
| 1 | [01-ml/README.md](01-ml/README.md) | How the model was built and evaluated (notebook `ser_report_notebook.ipynb`) |
| 1a | [01-ml/01-data-and-labels.md](01-ml/01-data-and-labels.md) | Configuration, the four datasets, label harmonisation, speaker IDs, class imbalance |
| 1b | [01-ml/02-preprocessing-augmentation.md](01-ml/02-preprocessing-augmentation.md) | Cleaning audio, fixed 3 s windows, augmentation, the `Dataset` |
| 1c | [01-ml/03-models.md](01-ml/03-models.md) | The CNN-BiLSTM-Attention model and the self-supervised transformer model |
| 1d | [01-ml/04-training-evaluation.md](01-ml/04-training-evaluation.md) | Training loop, the two evaluation protocols, metrics, analysis |
| 1e | [01-ml/05-export-and-realtime.md](01-ml/05-export-and-realtime.md) | ONNX export, int8 quantisation, voice-activity detection, sliding window |
| 2 | [02-backend/README.md](02-backend/README.md) | The FastAPI service (`backend/app.py`), its Dockerfile and libraries |
| 3 | [03-deployment/README.md](03-deployment/README.md) | Model → GitHub → Render → UptimeRobot → testing, and why each step |
| 4 | [04-android/README.md](04-android/README.md) | The Android app: recorder, on-device login, API client, wellness messages |

## The system on one page

```
                        TRAINING (done once, on Kaggle GPU)
   ┌───────────────┐   ┌────────────────┐   ┌────────────────┐   ┌─────────────────┐
   │ 4 public      │──▶│ pre-process +  │──▶│ train 2 kinds  │──▶│ evaluate with   │
   │ speech datasets│  │ augment        │   │ of model       │   │ 2 protocols     │
   └───────────────┘   └────────────────┘   └────────────────┘   └────────┬────────┘
                                                                          │ best model
                                                                          ▼
                                                                  export to ONNX file
                                                                  (model.onnx + labels.json)
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
        ▼
  GitHub repo ──▶ Render (Docker, free) runs  backend/app.py  (FastAPI + ONNX Runtime)
                                  ▲                      │
                 UptimeRobot pings /health               │ HTTPS
                 (keeps the free server awake)           ▼
                                           ┌──────────────────────────────┐
                                           │  Android app (Kotlin/Compose) │
                                           │  login → record 15 s → upload │
                                           │  → emotion + wellness message │
                                           └──────────────────────────────┘
```

### What happens when the user taps the microphone button (the whole story in 9 steps)

1. **Android** asks for the microphone permission (first time only) and records 16 kHz mono audio (`WavRecorder.kt`).
2. The app wraps the samples into a **WAV** file in memory and sends it with an HTTPS `POST /predict` (`EmotionApi.kt`).
3. **Render** wakes the container if it was asleep, and **FastAPI** receives the file (`backend/app.py`).
4. The server **decodes** the audio and converts it to 16 kHz mono (`load_audio`).
5. It cuts the audio into **3-second windows** (`windows`) and removes silence at the edges of each (`to_fixed`).
6. **ONNX Runtime** runs the neural network on every window (`SESSION.run`) and returns 7 numbers = scores for *angry … surprise*.
7. The scores are turned into **probabilities** (`softmax`) and averaged over the windows.
8. The server answers with JSON: best emotion, confidence, all probabilities, per-window results.
9. The app shows the emotion, the **wellness message** for that emotion (`Theme.kt`), and the score bars.

## Why these big decisions (the 10 answers a mentor will ask for)

| Decision | Why | Alternative we rejected |
|---|---|---|
| **7 emotions, 4 datasets** | One label set across corpora gives ~12 k clips and 121 speakers → more diversity than any single corpus | Using one dataset only: too few speakers, results not trustworthy |
| **"calm" merged into "neutral"** | Only RAVDESS has *calm*; keeping it would create a class with one source | Dropping calm clips: loses data; keeping it: class exists in 1 of 4 corpora |
| **Augmentation after the split, training only** | Otherwise slightly changed copies of a test clip end up in training = *data leakage*, inflated accuracy | Augmenting the whole dataset first (common mistake) |
| **Two evaluation protocols** | Random split shows how the model does on speakers it already heard; GroupKFold shows how it does on a **new person** – the real use case | Reporting only the random-split number (looks better, but misleading) |
| **Self-supervised transformers (Wav2Vec2/HuBERT/WavLM)** | Pre-trained on huge unlabeled speech, so a small labelled set is enough; best accuracy | Training from scratch: needs far more data |
| **A CNN-BiLSTM-Attention baseline too** | Small (8 MB) model that fits free hosting; shows the benefit of transformers by comparison | Skipping a baseline: no evidence that the big model is worth its cost |
| **ONNX + ONNX Runtime** | Run the model on a tiny CPU server **without PyTorch** (smaller, faster to start) | Shipping PyTorch: > 700 MB, does not fit a free server |
| **int8 quantisation (MatMul only)** | Model 4× smaller to fit the 512 MB free server; only the transformer's matrix multiplies are quantised to protect accuracy | Quantising everything: hurts the convolutional front-end |
| **FastAPI on Render (free) + UptimeRobot** | Free, no credit card, HTTPS built in; the monitor stops the 15-minute sleep | Hugging Face Docker Spaces (now paid), Heroku (no free tier) |
| **Login stored on the phone** | Needs no database or server code; the API stays simple | Server-side accounts: needs a persistent database (free hosts have none) |

## Glossary (short, in plain words)

| Term | Meaning |
|---|---|
| **Sample rate (16 kHz)** | How many audio measurements per second. 16 000 numbers per second is enough for speech and is what the pretrained models expect. |
| **Waveform** | The raw audio as a list of numbers over time. |
| **Spectrogram / mel spectrogram** | A picture of how much energy is at each pitch over time; *mel* spaces the pitches the way human hearing does. |
| **CNN** | Convolutional neural network – detects small local patterns (like edges in images, formant movements in spectrograms). |
| **BiLSTM** | A network that reads a sequence forwards and backwards to understand how it changes over time. |
| **Attention** | A learned weighting that lets the model focus on the most informative moments. |
| **Self-supervised model** | A network first trained on lots of *unlabeled* audio to predict hidden parts of it; later fine-tuned on a small labelled set. |
| **Transformer layer** | A building block that lets every time step look at every other (self-attention). |
| **Fine-tuning** | Continuing to train a pre-trained model on your own data with a small learning rate. |
| **Class weights / label smoothing** | Two tricks: give rare classes more weight in the loss; avoid over-confident targets. |
| **Data leakage** | Information from the test set sneaking into training, which makes accuracy look better than it is. |
| **GroupKFold** | Cross-validation where all clips of one speaker stay in the same fold, so test speakers are always unseen. |
| **Accuracy / macro-F1** | Share of correct predictions / average F1 over classes (rare classes count equally). |
| **ONNX** | A portable file format for neural networks. |
| **Quantisation (int8)** | Storing weights as 8-bit integers instead of 32-bit floats: smaller and faster. |
| **VAD** | Voice-activity detection: finds where someone is actually speaking. |
| **API / endpoint** | A web address the app calls; `POST /predict` is our endpoint. |
| **Docker** | A packaged mini-computer (image) so the server runs identically everywhere. |
| **Cold start** | The delay when a free server that went to sleep has to boot again. |
| **Edge-to-edge** | Android 15/16 draw apps behind the status and navigation bars, so apps must add padding. |

## Honest limitations to state yourself (it makes the project look rigorous)

* **Acted speech.** The datasets are performed by actors; real-life emotion is subtler, so real-world accuracy is lower.
* **Random-split vs unseen-speaker gap.** In the earlier Kaggle run the Wav2Vec2 model scored **86.5 %** on a random split but **76.6 %** on unseen speakers. The unseen-speaker number is the honest one. (These are measured values from the earlier run of the pipeline; the final report notebook prints its own numbers when you run it.)
* **Open API.** Anyone who knows the URL can call `/predict`; the app's login only protects the app screen.
* **Free-tier latency.** A request can take several seconds, and the first one after a long idle time up to a minute.
* **Not a medical device.** The wellness messages are general self-care suggestions, shown with a disclaimer.
