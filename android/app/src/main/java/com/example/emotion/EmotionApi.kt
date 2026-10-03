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

data class Prediction(
    val emotion: String?,
    val confidence: Double,
    val probabilities: List<Pair<String, Double>>,
    val speechDetected: Boolean,
    val latencyMs: Double,
)

/** Thin client for the FastAPI service in /backend. */
class EmotionApi(private val baseUrl: String) {
    // Free hosts sleep when idle: the first request can take up to ~a minute while the container wakes up.
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    suspend fun predict(wav: ByteArray, retries: Int = 2): Prediction = withContext(Dispatchers.IO) {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("file", "recording.wav", wav.toRequestBody("audio/wav".toMediaType()))
            .build()
        val req = Request.Builder().url(baseUrl.trimEnd('/') + "/predict").post(body).build()
        var last: Exception? = null
        repeat(retries + 1) { attempt ->
            try {
                client.newCall(req).execute().use { r ->
                    val text = r.body?.string().orEmpty()
                    if (!r.isSuccessful) throw IOException("Server error ${r.code}: ${text.take(200)}")
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
        val list = probs?.keys()?.asSequence()?.map { it to probs.getDouble(it) }?.sortedByDescending { it.second }?.toList()
            ?: emptyList()
        return Prediction(
            emotion = if (j.isNull("emotion")) null else j.getString("emotion"),
            confidence = j.optDouble("confidence", 0.0),
            probabilities = list,
            speechDetected = j.optBoolean("speech_detected", false),
            latencyMs = j.optDouble("latency_ms", 0.0),
        )
    }
}
