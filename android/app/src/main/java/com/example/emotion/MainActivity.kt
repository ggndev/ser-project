package com.example.emotion

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch

/** Replace with your Space URL, e.g. https://<user>-ser-api.hf.space (editable in the app too). */
private const val DEFAULT_BASE_URL = "https://YOUR-SERVICE.onrender.com"

private val EMOJI = mapOf(
    "angry" to "😠", "disgust" to "🤢", "fear" to "😨", "happy" to "😄",
    "neutral" to "😐", "sad" to "😢", "surprise" to "😲",
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { Surface(Modifier.fillMaxSize()) { EmotionScreen() } } }
    }
}

private sealed interface UiState {
    data object Idle : UiState
    data class Recording(val progress: Float) : UiState
    data object Analysing : UiState
    data class Done(val p: Prediction) : UiState
    data class Error(val msg: String) : UiState
}

@Composable
fun EmotionScreen() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val prefs = remember { ctx.getSharedPreferences("settings", Context.MODE_PRIVATE) }
    var baseUrl by remember { mutableStateOf(prefs.getString("base_url", DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL) }
    var state by remember { mutableStateOf<UiState>(UiState.Idle) }

    fun start() {
        scope.launch {
            try {
                prefs.edit().putString("base_url", baseUrl).apply()
                state = UiState.Recording(0f)
                val wav = WavRecorder.record(3) { state = UiState.Recording(it) }
                state = UiState.Analysing
                state = UiState.Done(EmotionApi(baseUrl).predict(wav))
            } catch (e: Exception) {
                state = UiState.Error(e.message ?: e.javaClass.simpleName)
            }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) start() else state = UiState.Error("Microphone permission is required.")
    }
    fun onRecordClicked() {
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) start()
        else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    Column(
        Modifier.fillMaxSize().padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Speech Emotion Recognizer", fontSize = 22.sp, fontWeight = FontWeight.Bold)
        OutlinedTextField(
            value = baseUrl, onValueChange = { baseUrl = it }, label = { Text("API base URL") },
            singleLine = true, modifier = Modifier.fillMaxWidth(),
        )
        val busy = state is UiState.Recording || state is UiState.Analysing
        Button(onClick = ::onRecordClicked, enabled = !busy, modifier = Modifier.fillMaxWidth().height(56.dp)) {
            Text(if (busy) "Please wait…" else "🎙️  Record 3 s & analyse")
        }
        when (val s = state) {
            UiState.Idle -> Text("Tap the button and speak for three seconds.")
            is UiState.Recording -> {
                Text("Recording… speak now")
                LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth())
            }
            UiState.Analysing -> {
                CircularProgressIndicator()
                Text("Analysing (the first request may take a minute while the free server wakes up)")
            }
            is UiState.Error -> Text(s.msg, color = MaterialTheme.colorScheme.error)
            is UiState.Done -> ResultCard(s.p)
        }
    }
}

@Composable
private fun ResultCard(p: Prediction) {
    if (!p.speechDetected || p.emotion == null) {
        Text("No speech detected — try again closer to the microphone.")
        return
    }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(EMOJI[p.emotion] ?: "", fontSize = 64.sp)
            Text(p.emotion.replaceFirstChar { it.uppercase() }, fontSize = 28.sp, fontWeight = FontWeight.Bold)
            Text("Confidence ${"%.1f".format(p.confidence * 100)}%  ·  ${p.latencyMs.toInt()} ms on server")
            Spacer(Modifier.height(4.dp))
            p.probabilities.forEach { (label, prob) ->
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(label, Modifier.width(80.dp))
                    LinearProgressIndicator(progress = { prob.toFloat() }, modifier = Modifier.weight(1f))
                    Text("%.0f%%".format(prob * 100), Modifier.width(48.dp), textAlign = androidx.compose.ui.text.style.TextAlign.End)
                }
            }
        }
    }
}
