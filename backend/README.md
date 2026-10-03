# Speech Emotion Recognition API (FastAPI + ONNX Runtime)

Lightweight, CPU-only, no PyTorch/librosa. See ../DEPLOY.md for free hosting on Render.

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000          # needs backend/model/model.onnx + labels.json
curl -F "file=@sample.wav" http://localhost:8000/predict
```
`POST /predict` (multipart field `file`, wav/flac/ogg/mp3, any rate; windows of 3 s with 50 % overlap are averaged) →
`{"emotion","confidence","probabilities","windows","speech_detected","latency_ms"}`. `GET /health` → status.
Env: `MODEL_FILE` (default `model.onnx`), `MAX_SECONDS` (30), `MIN_RMS` (0.003 silence gate).
