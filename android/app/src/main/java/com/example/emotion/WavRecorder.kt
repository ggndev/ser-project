package com.example.emotion

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Records 16 kHz mono 16-bit PCM from the microphone and wraps it in a WAV container (what the API expects). */
object WavRecorder {
    const val SAMPLE_RATE = 16_000

    /** Needs RECORD_AUDIO to be granted before calling. Returns the complete .wav file bytes. */
    @SuppressLint("MissingPermission")
    suspend fun record(seconds: Int = 3, onProgress: (Float) -> Unit = {}): ByteArray = withContext(Dispatchers.IO) {
        val total = SAMPLE_RATE * seconds
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, 4096) * 2
        )
        check(rec.state == AudioRecord.STATE_INITIALIZED) { "Microphone could not be initialised" }
        val pcm = ShortArray(total)
        try {
            rec.startRecording()
            var read = 0
            while (read < total) {
                val n = rec.read(pcm, read, minOf(2048, total - read))
                if (n <= 0) error("AudioRecord read failed ($n)")
                read += n
                onProgress(read.toFloat() / total)
            }
        } finally {
            runCatching { rec.stop() }
            rec.release()
        }
        toWav(pcm)
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
