package com.example.emotion

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/** Records 16 kHz mono 16-bit PCM from the microphone and wraps it in a WAV container (what the API expects). */
object WavRecorder {
    const val SAMPLE_RATE = 16_000
    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * Records until [stop] becomes true (after at least [minSeconds]) or [maxSeconds] is reached.
     * RECORD_AUDIO must already be granted. [onSeconds] is always called on the main thread (about 4x per second),
     * so the UI can be updated safely. Throws a readable exception if the microphone cannot be opened.
     */
    @SuppressLint("MissingPermission")
    suspend fun record(
        maxSeconds: Int,
        stop: AtomicBoolean,
        minSeconds: Int = 1,
        onSeconds: (Float) -> Unit = {},
    ): ByteArray = withContext(Dispatchers.IO) {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minBuf <= 0) throw IllegalStateException("This device does not support 16 kHz recording (buffer size $minBuf)")

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, minBuf * 4,
        )
        try {
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                throw IllegalStateException("Microphone could not be opened (is another app using it?)")
            }
            val total = SAMPLE_RATE * maxSeconds
            val pcm = ShortArray(total)
            var read = 0
            var lastReported = -1f
            recorder.startRecording()
            if (recorder.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                throw IllegalStateException("Microphone is busy or blocked by the system")
            }
            while (read < total && !(stop.get() && read >= SAMPLE_RATE * minSeconds)) {
                val n = recorder.read(pcm, read, minOf(2048, total - read))
                if (n < 0) throw IllegalStateException("Microphone read failed (error $n)")
                read += n
                val seconds = read.toFloat() / SAMPLE_RATE
                if (seconds - lastReported >= 0.25f) {
                    lastReported = seconds
                    mainHandler.post { onSeconds(seconds) }
                }
            }
            toWav(pcm.copyOf(read))
        } finally {
            try { recorder.stop() } catch (_: Throwable) { }
            recorder.release()
        }
    }

    private fun toWav(pcm: ShortArray): ByteArray {
        val dataLen = pcm.size * 2
        val bb = ByteBuffer.allocate(44 + dataLen).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()).putInt(36 + dataLen).put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1)              // PCM, mono
        bb.putInt(SAMPLE_RATE).putInt(SAMPLE_RATE * 2).putShort(2).putShort(16)      // rate, byte rate, align, bits
        bb.put("data".toByteArray()).putInt(dataLen)
        pcm.forEach { bb.putShort(it) }
        return bb.array()
    }
}
