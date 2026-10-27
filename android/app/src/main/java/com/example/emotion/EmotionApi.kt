package com.example.emotion

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

data class WindowResult(val startS: Double, val emotion: String?, val confidence: Double)

data class Prediction(
    val emotion: String?,
    val confidence: Double,
    val probabilities: List<Pair<String, Double>>,
    val speechDetected: Boolean,
    val latencyMs: Double,
    val windows: List<WindowResult> = emptyList(),
)

/** Thin client for the FastAPI service in /backend. */
class EmotionApi(private val baseUrl: String) {
    // Free hosts sleep when idle: the first request can take up to ~a minute while the container wakes up.
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private fun url(path: String) = baseUrl.trim().trimEnd('/') + path

    /** Round-trip time of GET /health in ms, or null if the server is unreachable (it may be waking up). */
    suspend fun health(): Long? = withContext(Dispatchers.IO) {
        try {
            val req = Request.Builder().url(url("/health")).get().build()
            val t0 = System.nanoTime()
            client.newCall(req).execute().use { r -> if (r.isSuccessful) (System.nanoTime() - t0) / 1_000_000 else null }
        } catch (e: Exception) {
            null
        }
    }

    /** POST /predict with the recorded WAV. Retries network errors (cold start). */
    suspend fun predict(wav: ByteArray, retries: Int = 2): Prediction = withContext(Dispatchers.IO) {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("file", "recording.wav", wav.toRequestBody("audio/wav".toMediaType()))
            .build()
        val req = Request.Builder().url(url("/predict")).post(body).build()
        var last: IOException? = null
        for (attempt in 0..retries) {
            try {
                client.newCall(req).execute().use { r ->
                    val text = r.body?.string().orEmpty()
                    if (!r.isSuccessful) throw IOException("Server error ${r.code}: ${text.take(160)}")
                    return@withContext parse(JSONObject(text))
                }
            } catch (e: IOException) {
                last = e
                if (attempt < retries) delay(2_000L * (attempt + 1))   // back off, then retry (cold start)
            }
        }
        throw last ?: IOException("request failed")
    }

    private fun parse(j: JSONObject): Prediction {
        val probs = j.optJSONObject("probabilities")
        val list = ArrayList<Pair<String, Double>>()
        if (probs != null) {
            val keys = probs.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                list.add(k to probs.getDouble(k))
            }
            list.sortByDescending { it.second }
        }
        val windows = ArrayList<WindowResult>()
        val arr = j.optJSONArray("windows")
        if (arr != null) {
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                windows.add(
                    WindowResult(
                        o.optDouble("start_s", 0.0),
                        if (o.isNull("emotion")) null else o.optString("emotion"),
                        o.optDouble("confidence", 0.0),
                    )
                )
            }
        }
        return Prediction(
            emotion = if (j.isNull("emotion")) null else j.optString("emotion"),
            confidence = j.optDouble("confidence", 0.0),
            probabilities = list,
            speechDetected = j.optBoolean("speech_detected", false),
            latencyMs = j.optDouble("latency_ms", 0.0),
            windows = windows,
        )
    }
}
