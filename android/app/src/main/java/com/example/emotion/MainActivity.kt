package com.example.emotion

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.util.concurrent.atomic.AtomicBoolean

/** Default API (free Render service). It can be changed in the app's Settings screen (gear icon). */
const val DEFAULT_BASE_URL = "https://ser-api-r6mj.onrender.com"

/** Longest recording. The server scores 3 s windows (up to 5), so ~15 s is the practical maximum on the free tier. */
const val MAX_RECORD_SECONDS = 15

/** Set to true when the app leaves the screen so a running recording stops cleanly. */
val stopRecording = AtomicBoolean(false)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()      // Android 15/16 draw apps edge-to-edge; the root below adds the system-bar insets
        setContent {
            AppTheme { Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) { Root() } }
        }
    }

    override fun onStop() {
        stopRecording.set(true)
        super.onStop()
    }
}

private enum class Screen { Login, Register, Main, Settings }

/** Top-level navigation: crash report dialog, login / register, the recorder, and the Settings (server URL) screen. */
@Composable
fun Root() {
    val ctx = LocalContext.current
    val prefs = remember { ctx.getSharedPreferences("settings", android.content.Context.MODE_PRIVATE) }
    val auth = remember { LocalAuth(ctx) }

    var user by remember { mutableStateOf(auth.currentUser()) }
    var screen by remember { mutableStateOf(if (user != null) Screen.Main else Screen.Login) }
    var settingsReturn by remember { mutableStateOf(Screen.Login) }
    var baseUrl by remember {
        val saved = prefs.getString("base_url", null)
        // ignore addresses saved by older builds (placeholder / Hugging Face)
        mutableStateOf(if (saved == null || "hf.space" in saved || "YOUR-SERVICE" in saved) DEFAULT_BASE_URL else saved)
    }

    fun openSettings() { settingsReturn = screen; screen = Screen.Settings }

    // Predictive back (Android 14+): Settings returns to where it came from, Register returns to Login
    BackHandler(enabled = screen == Screen.Settings || screen == Screen.Register) {
        screen = if (screen == Screen.Settings) settingsReturn else Screen.Login
    }

    CrashReportDialog()

    // safeDrawingPadding keeps content clear of the status bar, navigation bar, camera cut-out and keyboard;
    // widthIn keeps the layout readable on tablets / foldables.
    Box(Modifier.fillMaxSize().safeDrawingPadding(), contentAlignment = Alignment.TopCenter) {
        Box(Modifier.widthIn(max = 600.dp).fillMaxSize()) {
            when (screen) {
                Screen.Login -> LoginScreen(
                    auth = auth,
                    onSignedIn = { user = auth.currentUser(); screen = Screen.Main },
                    onCreateAccount = { screen = Screen.Register },
                    onSettings = ::openSettings,
                )
                Screen.Register -> RegisterScreen(
                    auth = auth,
                    onRegistered = { user = auth.currentUser(); screen = Screen.Main },
                    onBack = { screen = Screen.Login },
                )
                Screen.Main -> MainScreen(
                    baseUrl = baseUrl,
                    user = user.orEmpty(),
                    onSettings = ::openSettings,
                    onSignOut = { auth.signOut(); user = null; screen = Screen.Login },
                )
                Screen.Settings -> SettingsScreen(
                    currentUrl = baseUrl,
                    onSave = { url -> baseUrl = url; prefs.edit().putString("base_url", url).apply(); screen = settingsReturn },
                    onBack = { screen = settingsReturn },
                )
            }
        }
    }
}
