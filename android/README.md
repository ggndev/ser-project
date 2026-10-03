# Android client (Kotlin + Jetpack Compose)

Records 3 s of audio (16 kHz mono WAV), POSTs it to the API's `/predict`, shows the emotion, confidence and per-class bars.

1. Open this `android/` folder in **Android Studio** (Koala or newer; it generates the Gradle wrapper on first sync).
2. Set `DEFAULT_BASE_URL` in `MainActivity.kt` to your Space URL (`https://gxgxn02-ser-api.hf.space`) — or just type it into the app's text field (remembered between runs).
3. See ../DEPLOY.md (section B) for phone / emulator steps.

The first request after the free Space has been idle can take up to ~1 minute; the client uses a 90 s read timeout and retries twice.
