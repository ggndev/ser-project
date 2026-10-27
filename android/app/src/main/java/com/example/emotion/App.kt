package com.example.emotion

import android.app.Application
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.util.Date

/**
 * Application class: installs a crash reporter. If the app ever crashes, the stack trace is saved to a file and shown
 * (with a Copy button) the next time the app opens, so the exact error can be sent back without any developer tools.
 */
class App : Application() {
    override fun onCreate() {
        super.onCreate()
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            try {
                File(filesDir, CrashReport.FILE).writeText(
                    "Time: ${Date()}\n" +
                        "Device: ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n" +
                        "Thread: ${thread.name}\n\n" + Log.getStackTraceString(error)
                )
            } catch (_: Throwable) {
                // never let the reporter itself crash
            }
            previous?.uncaughtException(thread, error)
        }
    }
}

object CrashReport {
    const val FILE = "last_crash.txt"

    fun read(context: Context): String? =
        try { File(context.filesDir, FILE).takeIf { it.exists() }?.readText()?.take(6000) } catch (_: Throwable) { null }

    fun clear(context: Context) {
        try { File(context.filesDir, FILE).delete() } catch (_: Throwable) { }
    }
}
