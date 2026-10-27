package com.example.emotion

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import kotlin.math.roundToInt

private const val TAG = "EmoSense"

private fun friendlyError(e: Throwable): String = when (e) {
    is SocketTimeoutException -> "The free server is waking up. Wait about 30 seconds and try again."
    is UnknownHostException -> "No internet connection (or the server address is wrong)."
    else -> e.message ?: e.javaClass.simpleName
}

// ================================================================== crash report (shown once after a crash)
@Composable
fun CrashReportDialog() {
    val ctx = LocalContext.current
    var crash by remember { mutableStateOf(CrashReport.read(ctx)) }
    val report = crash ?: return
    AlertDialog(
        onDismissRequest = { CrashReport.clear(ctx); crash = null },
        title = { Text("The app closed unexpectedly last time") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text("Please copy this report and send it to the developer:", fontSize = 13.sp)
                Spacer(Modifier.height(8.dp))
                SelectionContainer { Text(report, fontSize = 11.sp) }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("crash report", report))
            }) { Text("Copy") }
        },
        dismissButton = { TextButton(onClick = { CrashReport.clear(ctx); crash = null }) { Text("Close") } },
    )
}

// ================================================================== login
@Composable
fun LoginScreen(auth: LocalAuth, onSignedIn: () -> Unit, onCreateAccount: () -> Unit, onSettings: () -> Unit) {
    var userId by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), horizontalArrangement = Arrangement.End) {
            TextButton(onClick = onSettings) { Text("⚙ Settings") }
        }
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
        ) {
            Text("🎙️", fontSize = 56.sp)
            Text("EmoSense", fontSize = 30.sp, fontWeight = FontWeight.Bold)
            Text("Speech emotion & wellness check", color = Color.Gray, fontSize = 14.sp)
            Spacer(Modifier.height(28.dp))

            OutlinedTextField(userId, { userId = it }, label = { Text("User ID") }, singleLine = true, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(10.dp))
            OutlinedTextField(
                password, { password = it }, label = { Text("Password") }, singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(14.dp))
            error?.let {
                Text(it, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
                Spacer(Modifier.height(8.dp))
            }
            Button(
                onClick = {
                    val result = auth.login(userId, password)
                    if (result.ok) onSignedIn() else error = result.message
                },
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) { Text("Sign in", fontSize = 16.sp) }
            TextButton(onClick = onCreateAccount) { Text("New here? Create account") }
        }
    }
}

@Composable
fun RegisterScreen(auth: LocalAuth, onRegistered: () -> Unit, onBack: () -> Unit) {
    var userId by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirm by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp)) {
            TextButton(onClick = onBack) { Text("← Back") }
        }
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
        ) {
            Text("Create account", fontSize = 26.sp, fontWeight = FontWeight.Bold)
            Text("Your account is stored only on this phone.", color = Color.Gray, fontSize = 13.sp)
            Spacer(Modifier.height(24.dp))

            OutlinedTextField(userId, { userId = it }, label = { Text("User ID") }, singleLine = true, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(10.dp))
            OutlinedTextField(
                password, { password = it }, label = { Text("Password (min 6 characters)") }, singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password), modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(10.dp))
            OutlinedTextField(
                confirm, { confirm = it }, label = { Text("Confirm password") }, singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password), modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(14.dp))
            error?.let {
                Text(it, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
                Spacer(Modifier.height(8.dp))
            }
            Button(
                onClick = {
                    val result = auth.register(userId, password, confirm)
                    if (result.ok) onRegistered() else error = result.message
                },
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) { Text("Create account", fontSize = 16.sp) }
        }
    }
}

// ================================================================== settings (server address)
@Composable
fun SettingsScreen(currentUrl: String, onSave: (String) -> Unit, onBack: () -> Unit) {
    val scope = rememberCoroutineScope()
    var url by remember { mutableStateOf(currentUrl) }
    var status by remember { mutableStateOf<String?>(null) }
    var testing by remember { mutableStateOf(false) }

    fun cleaned(): String = url.trim().trimEnd('/')
    fun valid(): Boolean = (cleaned().startsWith("http://") || cleaned().startsWith("https://")) && cleaned().toHttpUrlOrNull() != null

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp)) {
            TextButton(onClick = onBack) { Text("← Back") }
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Settings", fontSize = 26.sp, fontWeight = FontWeight.Bold)
            Text("Server address", fontWeight = FontWeight.Medium)
            OutlinedTextField(
                url, { url = it; status = null }, singleLine = true, modifier = Modifier.fillMaxWidth(),
                label = { Text("https://…") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            )
            Text("This is the cloud service that analyses your voice. The free server sleeps when idle, so the first request can take up to a minute.", fontSize = 12.sp, color = Color.Gray)

            status?.let { Text(it, color = if (it.startsWith("✅")) Color(0xFF2E7D32) else MaterialTheme.colorScheme.error) }

            OutlinedButton(
                onClick = {
                    if (!valid()) { status = "Please enter a full address starting with http:// or https://"; return@OutlinedButton }
                    testing = true; status = "Testing… (the server may be waking up)"
                    scope.launch {
                        val ms = try { EmotionApi(cleaned()).health() } catch (e: CancellationException) { throw e } catch (e: Throwable) { null }
                        status = if (ms != null) "✅ Connected (${ms} ms)" else "Could not reach the server. Check the address or try again in a minute."
                        testing = false
                    }
                },
                enabled = !testing, modifier = Modifier.fillMaxWidth(),
            ) { Text(if (testing) "Testing…" else "Test connection") }

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = { url = DEFAULT_BASE_URL; status = null }, modifier = Modifier.weight(1f)) { Text("Reset to default") }
                Button(
                    onClick = { if (valid()) onSave(cleaned()) else status = "Please enter a full address starting with http:// or https://" },
                    modifier = Modifier.weight(1f),
                ) { Text("Save") }
            }
        }
    }
}

// ================================================================== main screen (recorder + result)
private sealed interface UiState {
    data object Idle : UiState
    data class Recording(val seconds: Float) : UiState
    data object Analysing : UiState
    data class Done(val p: Prediction) : UiState
    data class Error(val msg: String) : UiState
}

@Composable
fun MainScreen(baseUrl: String, user: String, onSettings: () -> Unit, onSignOut: () -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<UiState>(UiState.Idle) }
    var serverText by remember { mutableStateOf("Checking server…") }

    LaunchedEffect(baseUrl) {
        serverText = "Checking server…"
        val ms = EmotionApi(baseUrl).health()
        serverText = if (ms != null) "Server online" else "Server is waking up (the first request can take a minute)"
    }

    fun startRecording() {
        if (state is UiState.Recording || state is UiState.Analysing) return
        scope.launch {
            try {
                stopRecording.set(false)
                state = UiState.Recording(0f)
                val wav = WavRecorder.record(maxSeconds = MAX_RECORD_SECONDS, stop = stopRecording) { sec ->
                    if (state is UiState.Recording) state = UiState.Recording(sec)   // ignore late callbacks
                }
                state = UiState.Analysing
                state = UiState.Done(EmotionApi(baseUrl).predict(wav))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                Log.e(TAG, "recording/analysis failed", e)
                state = UiState.Error(friendlyError(e))
            }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startRecording()
        else state = UiState.Error("The microphone permission is needed to hear your voice. You can allow it in Settings ▸ Apps ▸ EmoSense ▸ Permissions.")
    }

    fun onButtonClicked() {
        val s = state
        if (s is UiState.Recording) {
            stopRecording.set(true)
        } else if (s !is UiState.Analysing) {
            if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording()
            else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("EmoSense", fontSize = 26.sp, fontWeight = FontWeight.Bold)
                Text("Signed in as $user", fontSize = 12.sp, color = Color.Gray)
            }
            TextButton(onClick = onSettings) { Text("⚙") }
            TextButton(onClick = onSignOut) { Text("Sign out") }
        }
        Text(serverText, fontSize = 12.sp, color = Color.Gray)

        val recording = state is UiState.Recording
        val analysing = state is UiState.Analysing
        Button(
            onClick = ::onButtonClicked,
            enabled = !analysing,
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(containerColor = if (recording) Color(0xFFE5484D) else BrandColor),
            modifier = Modifier.size(150.dp),
        ) { Text(if (recording) "⏹" else "🎙️", fontSize = 48.sp) }

        when (val s = state) {
            UiState.Idle, is UiState.Error, is UiState.Done ->
                Text("Tap the button and speak for up to $MAX_RECORD_SECONDS seconds, then tap again to stop.", textAlign = TextAlign.Center, color = Color.Gray)
            is UiState.Recording -> {
                Text("Listening… ${"%.1f".format(s.seconds)} s  (tap to stop)", fontWeight = FontWeight.Medium)
                LinearProgressIndicator(progress = { (s.seconds / MAX_RECORD_SECONDS).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
            }
            UiState.Analysing -> {
                CircularProgressIndicator()
                Text("Analysing your voice… this can take a few seconds.", textAlign = TextAlign.Center)
            }
        }

        when (val s = state) {
            is UiState.Done -> ResultCard(s.p)
            is UiState.Error -> InfoCard("⚠️", "Something went wrong", s.msg)
            else -> {}
        }
    }
}

@Composable
private fun InfoCard(emoji: String, title: String, body: String) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFF3F4F8))) {
        Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(emoji, fontSize = 36.sp)
            Text(title, fontWeight = FontWeight.Bold, fontSize = 17.sp)
            Text(body, textAlign = TextAlign.Center, color = Color.DarkGray)
        }
    }
}

@Composable
private fun ResultCard(p: Prediction) {
    val emotion = p.emotion
    if (!p.speechDetected || emotion == null) {
        InfoCard("🤫", "I could not hear any speech", "Move closer to the microphone and try again.")
        return
    }
    val st = styleOf(emotion)
    val timeline = p.windows.mapNotNull { it.emotion }
    val changed = timeline.distinct().size > 1

    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFF7F7FB))) {
        Column(Modifier.padding(18.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(st.emoji, fontSize = 64.sp)
            Text(st.title, fontSize = 28.sp, fontWeight = FontWeight.Bold, color = st.color)
            Text("Confidence ${(p.confidence * 100).roundToInt()}%", color = Color.DarkGray)

            if (p.confidence < 0.5) {
                Text(LOW_CONFIDENCE_NOTE, fontSize = 13.sp, color = Color(0xFFB26A00), textAlign = TextAlign.Center)
            }

            // ---- wellness guidance for the detected emotion
            Column(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(st.color.copy(alpha = 0.12f)).padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(st.headline, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                Text(st.insight, fontSize = 14.sp, lineHeight = 20.sp)
                st.tips.forEach { tip -> Text("•  $tip", fontSize = 14.sp, lineHeight = 20.sp) }
                if (emotion == "sad") Text(SADNESS_NOTE, fontSize = 13.sp, lineHeight = 18.sp, fontWeight = FontWeight.Medium)
            }
            if (changed) {
                Text(
                    "Your tone changed during the recording: " + timeline.joinToString(" → ") { styleOf(it).emoji },
                    fontSize = 14.sp, textAlign = TextAlign.Center,
                )
            }
            Text("$DISCLAIMER $SAFETY_NOTE", fontSize = 11.sp, color = Color.Gray, textAlign = TextAlign.Center)

            // ---- all emotion scores
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                p.probabilities.forEach { (label, prob) ->
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text("${styleOf(label).emoji} ${styleOf(label).title}", Modifier.width(110.dp), fontSize = 13.sp)
                        LinearProgressIndicator(
                            progress = { prob.toFloat().coerceIn(0f, 1f) },
                            modifier = Modifier.weight(1f).height(8.dp).clip(RoundedCornerShape(4.dp)),
                            color = styleOf(label).color,
                        )
                        Text("${(prob * 100).roundToInt()}%", Modifier.width(44.dp), fontSize = 12.sp, textAlign = TextAlign.End)
                    }
                }
            }
        }
    }
}
