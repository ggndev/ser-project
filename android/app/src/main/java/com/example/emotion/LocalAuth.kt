package com.example.emotion

import android.content.Context
import java.security.MessageDigest
import java.security.SecureRandom

/** Outcome of a register / login attempt. */
class AuthResult(val ok: Boolean, val message: String = "")

/**
 * User ID + password accounts that live ONLY on this phone (no server involved).
 *
 * - Passwords are never stored: we keep a random 16-byte salt and an iterated SHA-256 hash of salt+password.
 * - User IDs are case-insensitive and unique per device.
 * - After 5 wrong passwords the login is locked for 30 seconds.
 * - The signed-in user is remembered until [signOut].
 *
 * Note: this protects the app screen, not the cloud API. Uninstalling the app or clearing its data deletes the accounts.
 */
class LocalAuth(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("auth", Context.MODE_PRIVATE)

    fun currentUser(): String? = prefs.getString(KEY_SESSION, null)

    fun register(userId: String, password: String, confirm: String): AuthResult {
        val id = userId.trim()
        if (!ID_RE.matches(id)) return AuthResult(false, "User ID must be 3–20 characters: letters, digits, dot, dash or underscore.")
        if (password.length < MIN_PASSWORD) return AuthResult(false, "Password must be at least $MIN_PASSWORD characters.")
        if (password != confirm) return AuthResult(false, "The two passwords do not match.")
        if (prefs.contains(hashKey(id))) return AuthResult(false, "This User ID already exists on this phone. Please sign in.")
        val salt = ByteArray(16).also { SecureRandom().nextBytes(it) }
        prefs.edit()
            .putString(hashKey(id), toHex(salt) + ":" + toHex(hash(password, salt)))
            .putString(nameKey(id), id)
            .putString(KEY_SESSION, id)
            .apply()
        return AuthResult(true)
    }

    fun login(userId: String, password: String): AuthResult {
        val now = System.currentTimeMillis()
        val lockedUntil = prefs.getLong(KEY_LOCKED_UNTIL, 0L)
        if (now < lockedUntil) return AuthResult(false, "Too many attempts. Try again in ${(lockedUntil - now + 999) / 1000} s.")

        val id = userId.trim()
        val stored = prefs.getString(hashKey(id), null)
            ?: return AuthResult(false, "No account with this User ID on this phone. Tap “Create account”.")
        if (!verify(password, stored)) {
            val fails = prefs.getInt(KEY_FAILS, 0) + 1
            val e = prefs.edit()
            if (fails >= MAX_FAILS) e.putInt(KEY_FAILS, 0).putLong(KEY_LOCKED_UNTIL, now + LOCK_MS) else e.putInt(KEY_FAILS, fails)
            e.apply()
            return AuthResult(false, "Wrong password.")
        }
        prefs.edit().putInt(KEY_FAILS, 0).putString(KEY_SESSION, prefs.getString(nameKey(id), id)).apply()
        return AuthResult(true)
    }

    fun signOut() {
        prefs.edit().remove(KEY_SESSION).apply()
    }

    // ---------------------------------------------------------------- helpers
    private fun hashKey(id: String) = "hash_" + id.lowercase()
    private fun nameKey(id: String) = "name_" + id.lowercase()

    private fun hash(password: String, salt: ByteArray): ByteArray {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(salt)
        var d = md.digest(password.toByteArray(Charsets.UTF_8))
        repeat(ITERATIONS) {
            md.update(d)
            md.update(salt)
            d = md.digest()
        }
        return d
    }

    private fun verify(password: String, stored: String): Boolean {
        val parts = stored.split(":")
        if (parts.size != 2) return false
        val salt = fromHex(parts[0]) ?: return false
        val expected = fromHex(parts[1]) ?: return false
        return MessageDigest.isEqual(expected, hash(password, salt))      // constant-time comparison
    }

    private fun toHex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }

    private fun fromHex(s: String): ByteArray? {
        if (s.length % 2 != 0) return null
        return try { ByteArray(s.length / 2) { s.substring(it * 2, it * 2 + 2).toInt(16).toByte() } } catch (e: NumberFormatException) { null }
    }

    companion object {
        private val ID_RE = Regex("^[A-Za-z0-9._-]{3,20}$")
        private const val MIN_PASSWORD = 6
        private const val ITERATIONS = 20_000
        private const val MAX_FAILS = 5
        private const val LOCK_MS = 30_000L
        private const val KEY_SESSION = "signed_in_user"
        private const val KEY_FAILS = "fails"
        private const val KEY_LOCKED_UNTIL = "locked_until"
    }
}
