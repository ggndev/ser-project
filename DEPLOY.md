# Deploy & run (100 % free, no credit card)

Full explanations: see [docs/03-deployment/README.md](docs/03-deployment/README.md). Quick checklist:

1. **Model files** → `backend/model/model.onnx` (the int8 Wav2Vec2 export, ≈ 95 MB) and `backend/model/labels.json`. Smaller fallback: `backend/model_cnn/` (8 MB CNN-BiLSTM-Attention).
2. **GitHub** → push the project to `ggndev/ser-project` (use `git push`; the browser upload limit is 25 MB per file).
3. **Render** → New ▸ Web Service ▸ repo ▸ Language **Docker** ▸ **Root Directory `backend`** ▸ Dockerfile path `./Dockerfile` ▸ Instance **Free**. URL: `https://ser-api-r6mj.onrender.com`.
4. **Check** → open `…/health` and `…/docs`.
5. **Keep awake** → free UptimeRobot HTTP monitor on `…/health`, every 5 minutes.
6. **Test** → import `postman/SER_API.postman_collection.json` (Postman or Insomnia); working directory = the `postman` folder.
7. **Android** → install `apk/EmoSense-release.apk` (≈ 1 MB) or build the `android/` project in Android Studio. Create an account in the app; the server address is under ⚙ Settings.

Login (user ID + password) is handled **inside the app**; the API is open.
If the server runs out of memory, copy `backend/model_cnn/model.onnx` and `labels.json` over `backend/model/` and push again.
