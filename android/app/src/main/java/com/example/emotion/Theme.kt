package com.example.emotion

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** Look and wellness guidance for one detected emotion. Wording is supportive and non-clinical on purpose. */
data class EmotionStyle(
    val color: Color,
    val emoji: String,
    val title: String,
    val headline: String,       // one-line reading of the voice
    val insight: String,        // short, kind explanation
    val tips: List<String>,     // three small, doable actions
)

private val STYLES = mapOf(
    "angry" to EmotionStyle(
        Color(0xFFE5484D), "😠", "Angry",
        "Your voice sounds tense",
        "Strong feelings are normal. A short pause before reacting usually helps a lot.",
        listOf(
            "Breathe in for 4 seconds, out for 6 seconds, five times.",
            "Step away for a few minutes — a short walk or a glass of cold water.",
            "Write down what triggered it before you reply to anyone.",
        ),
    ),
    "disgust" to EmotionStyle(
        Color(0xFF7CB342), "🤢", "Disgust",
        "Something feels off-putting",
        "Your voice suggests you are reacting strongly against something. Give yourself some distance from it.",
        listOf(
            "Take a short break and get some fresh air.",
            "Name what is bothering you, in one sentence.",
            "Reset your senses: sip some water or wash your face.",
        ),
    ),
    "fear" to EmotionStyle(
        Color(0xFF8E5BD9), "😨", "Fear",
        "You may be feeling anxious",
        "Worry can make everything feel urgent. Slowing down your body helps your mind follow.",
        listOf(
            "Try 5-4-3-2-1: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
            "Box breathing: in 4, hold 4, out 4, hold 4 — repeat four times.",
            "Tell someone you trust what is worrying you.",
        ),
    ),
    "happy" to EmotionStyle(
        Color(0xFFF5A300), "😄", "Happy",
        "You sound upbeat",
        "Lovely! Positive moments are worth noticing — they build resilience.",
        listOf(
            "Notice what is making you feel good right now.",
            "Share the good feeling with a friend or family member.",
            "Write one thing you are grateful for today.",
        ),
    ),
    "neutral" to EmotionStyle(
        Color(0xFF6B7F92), "😐", "Neutral",
        "You sound calm and steady",
        "A balanced moment is a good time to look after your body.",
        listOf(
            "Stretch your neck, shoulders and back for a minute.",
            "Drink a glass of water.",
            "Take one mindful minute: just notice your breathing.",
        ),
    ),
    "sad" to EmotionStyle(
        Color(0xFF3B82F6), "😢", "Sad",
        "You may be feeling low",
        "It is okay to feel this way. Be gentle with yourself — small steps count.",
        listOf(
            "Take three slow, deep breaths.",
            "Reach out to a friend or family member, even with a short message.",
            "Get some daylight or a little light movement, like a short walk.",
        ),
    ),
    "surprise" to EmotionStyle(
        Color(0xFFE0559C), "😲", "Surprise",
        "Something unexpected?",
        "Surprises can feel exciting or unsettling. Give yourself a moment to process it.",
        listOf(
            "Pause and take a few slow breaths.",
            "Name the feeling: is it excitement, worry, or both?",
            "Give yourself some time before you react or decide.",
        ),
    ),
)

val DefaultStyle = EmotionStyle(Color(0xFF5B5BD6), "🎙️", "Ready", "", "", emptyList())
val BrandColor = Color(0xFF5B5BD6)

const val LOW_CONFIDENCE_NOTE = "I am not very sure about this reading. Try recording again, a little closer to the microphone, for a few seconds."
const val SADNESS_NOTE = "If this feeling stays heavy for more than two weeks, consider talking to a doctor or counsellor."
const val SAFETY_NOTE = "If you ever feel unsafe or in crisis, contact your local emergency services or a trusted person right away."
const val DISCLAIMER = "This is an AI estimate from your voice, not a medical diagnosis."

fun styleOf(emotion: String?): EmotionStyle = STYLES[emotion] ?: DefaultStyle

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = lightColorScheme(primary = BrandColor), content = content)
}
