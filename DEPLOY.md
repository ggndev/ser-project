# Deploy (100 % free, no credit card) & run

## Why not the Wav2Vec2 model?
Free hosts give ~512 MB RAM. The fine-tuned Wav2Vec2 ONNX is 378 MB and needs ~700 MB at run time, and Hugging Face Docker Spaces are now paid. So the API serves the **lightweight CNN-BiLSTM-Attention** model (8 MB, ~3 ms per 3 s window, ~80 MB RAM) — the same architecture/pre-processing as in the notebook, trained speaker-independently. Your Wav2Vec2 files are parked in `backend/model_wav2vec2_unused/`.

## Option A (try first): your 94 MB `model_int8.onnx` (Wav2Vec2, quantised)
It is already copied to `backend/model/model.onnx` (+ `labels.json`). Measured here: ~340 MB RAM, ~0.5 s per window on a fast PC core. It fits Render's 512 MB, but Render's free CPU is a fraction of a core, so expect several seconds per request (the app waits up to 90 s). **Accuracy caveat:** quantisation can cost accuracy — look at the Kaggle export cell's printed line `onnx int8: … argmax agreement NN%` (int8 vs fp32 on real test clips). If agreement is below ~95 % or Render runs out of memory/too slow, use Option B.

## Option B – the lightweight CNN
### Step 1 – export the light model (once)
You need the checkpoint `best_p2_CNN-BiLSTM-Attn_full.pt` from the Kaggle run (notebook Output tab → `ser_work/`; it is ~5 MB).
```bash
# on this PC (needs torch+torchaudio; the repo venv has them) or in a Kaggle cell
python3 tools/export_cnn.py /path/to/best_p2_CNN-BiLSTM-Attn_full.pt backend/model
```
→ creates `backend/model/model.onnx` (≈8 MB) and `backend/model/labels.json`.
Please paste me the notebook's *comparison table* (CNN P1 / P2 numbers) so we know the accuracy that is being deployed.

## Step 2 – put the code on GitHub
Create a free GitHub repo (e.g. `ser-project`), push this folder (the 8 MB model is fine for normal git; `.gitignore` already excludes the big files).

## Step 3 – host on Render (free web service)
1. render.com → sign up with GitHub → **New ▸ Blueprint** (uses `render.yaml`) or **New ▸ Web Service** → pick the repo → Runtime **Docker**, Root directory `backend`, Instance type **Free**.
2. Wait for the deploy (~5 min). URL looks like `https://ser-api-xxxx.onrender.com`; check `…/health`.
   *If Render asks for a payment method on your account, stop and tell me — fallback is PythonAnywhere (also free, no card).*
3. **Avoid sleep/timeouts:** free services sleep after 15 min idle (≈1 min wake-up). Make a free UptimeRobot (uptimerobot.com) HTTP monitor on `…/health` every 5 min — keeps it awake. (750 free hours/month ≈ one always-on service.)

## Step 4 – Android app
1. Android Studio → Open `android/` → Sync.
2. Put your Render URL in `DEFAULT_BASE_URL` (`MainActivity.kt`) or type it in the app.
3. Phone: USB debugging → ▶ Run (or Build ▸ Build APK → copy `app-debug.apk`). Emulator: Pixel API 34, Extended controls ▸ Microphone ▸ host audio input.
4. Allow microphone → **Record 3 s & analyse**.
