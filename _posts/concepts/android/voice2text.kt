package com.example.voicetranscriber

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var textView: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var transcriptionFile: File
    private var isListening = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        textView = findViewById(R.id.textView)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)

        // Request necessary permissions
        requestPermissions()

        // Initialize speech recognizer
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        val recognitionListener = RecognitionListenerImpl()
        speechRecognizer.setRecognitionListener(recognitionListener)

        // Set up buttons
        startButton.setOnClickListener {
            if (isListening) return@setOnClickListener
            startListening()
        }

        stopButton.setOnClickListener {
            if (!isListening) return@setOnClickListener
            stopListening()
        }
    }

    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )

        val permissionGranted = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

        if (!permissionGranted) {
            ActivityCompat.requestPermissions(this, permissions, 1)
        }
    }

    private fun startListening() {
        isListening = true
        startButton.isEnabled = false
        stopButton.isEnabled = true

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)

        speechRecognizer.startListening(intent)
    }

    private fun stopListening() {
        isListening = false
        startButton.isEnabled = true
        stopButton.isEnabled = false
        speechRecognizer.stopListening()
    }

    private fun saveTranscription(text: String) {
        try {
            // Create file with timestamp
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            transcriptionFile = File(filesDir, "transcription_$timestamp.txt")

            // Write to file
            FileWriter(transcriptionFile).use { writer ->
                writer.write(text)
            }

            Toast.makeText(this, "Transcription saved to ${transcriptionFile.name}", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error saving transcription: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private inner class RecognitionListenerImpl : RecognitionListener {
        override fun onResults(results: Bundle?) {
            results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.let { result ->
                val transcription = result[0]
                textView.text = transcription
                saveTranscription(transcription)
            }
        }

        override fun onError(error: Int) {
            Toast.makeText(this@MainActivity, "Speech recognition error: $error", Toast.LENGTH_SHORT).show()
        }

        override fun onReadyForSpeech(params: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onPartialResults(partialResults: Bundle?) {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
    }
}