# 📱 Android app: EmoSense

A simple, lightweight Android app that **records your voice**, sends it to the Speech Emotion Recognition API and shows the detected emotion with a **caring wellness message**.

> Part of the [Speech Emotion Recognition project](../README.md). Detailed explanation: [docs/04-android](../docs/04-android/README.md).

## Features

- 🔐 **On-device accounts**: register with a user ID and password, sign in, stay signed in. Passwords are stored salted and hashed, and never leave the phone.
- 🎤 **Record up to 15 seconds** with a live timer; tap again to stop early.
- 🧠 **Emotion result** with confidence and a score bar for every emotion.
- 📈 **"Tone changed" timeline** for longer recordings.
- 💚 **Wellness messages** matched to the detected emotion, plus gentle notes for sadness and low confidence, and a clear not-a-diagnosis notice.
- ⚙️ **Settings screen** to enter your server address, test the connection and reset it.
- 🛟 **Crash reporter**: if the app ever closes unexpectedly, the next launch offers to show the error so it can be fixed.
- 📐 **Modern Android**: edge-to-edge screens, predictive back gesture, Android 16 target, small R8-shrunk release build.

## Screens

| Screen | What it does |
|---|---|
| Login | Sign in with user ID and password |
| Register | Create an account on the device |
| Main | Record, analyse and view the result |
| Settings (⚙) | Set and test the server address |

## Requirements

| Item | Value |
|---|---|
| Minimum Android | 7.0 (API 24) |
| Target Android | 16 (API 36) |
| Language / UI | Kotlin, Jetpack Compose (Material 3) |
| Networking | OkHttp, `org.json` |
| Build tools | Android Studio, **JDK 17**, Gradle wrapper included |
| Permissions | `INTERNET`, `RECORD_AUDIO` |

## Install the ready-made app

Download [`../apk/EmoSense-release.apk`](../apk/EmoSense-release.apk) on your phone, open it, and allow installation from that source when asked.

## Build from source

1. Open the **`android/`** folder in Android Studio and let Gradle sync (internet needed the first time).
2. Connect a phone with USB debugging, or use an emulator.
3. Press **Run**.

Or from a terminal inside `android/`:

```bash
./gradlew assembleDebug      # APK in app/build/outputs/apk/debug/
./gradlew assembleRelease    # smaller, shrunk build
```

If Gradle complains about Java, point `JAVA_HOME` to a JDK 17.

## Connect the app to your server

1. Deploy the backend (see [../backend/README.md](../backend/README.md)).
2. In the app, tap ⚙ **Settings**, enter your server address (for example `https://<your-service>.onrender.com`), tap **Test connection**, then **Save**.

The server address is kept in the Settings screen on purpose, so it is not shown on the login page.

## Project layout

```
android/app/src/main/java/com/example/emotion/
├── App.kt            crash reporter set-up
├── MainActivity.kt   navigation (Login, Register, Main, Settings), back handling
├── Screens.kt        all Compose screens and the result card
├── Theme.kt          colours, emojis and wellness text for each emotion
├── LocalAuth.kt      on-device register / sign-in (salted, hashed passwords)
├── WavRecorder.kt    16 kHz microphone recording and WAV encoding
└── EmotionApi.kt     HTTP client for /health and /predict with retries
```

## Troubleshooting

| Problem | What to do |
|---|---|
| "Could not reach the server" | Check the address in ⚙ Settings and use **Test connection**. A free server may need about a minute to wake up. |
| First analysis is slow | Free cloud CPUs are slow; wait for "Analysing…" to finish. |
| No microphone access | Allow the **Microphone** permission in the Android app settings. |
| App closed unexpectedly | Re-open it; a dialog shows the error details to share with the developers. |

## Privacy

Recordings are sent to the server only for analysis and are processed in memory. Accounts and settings stay on the phone.
