import speech_recognition as sr
import time
import datetime

def transcribe_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Set up microphone
    microphone = sr.Microphone()
    
    print("Initializing microphone...")
    
    # Create or open the sleep.txt file in append mode
    with open('sleep.txt', 'a') as log_file:
        # Adjust for ambient noise
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Microphone initialized. You can start speaking.")
            
        try:
            while True:
                print("\nListening...")
                
                # Listen for speech
                with microphone as source:
                    audio = recognizer.listen(source)
                    
                try:
                    # Convert speech to text
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    
                    # Log the text with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {text}\n"
                    log_file.write(log_entry)
                    log_file.flush()  # Ensure data is written to file immediately
                    
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                
                # Small delay before next listen
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nTranscription stopped by user")

if __name__ == "__main__":
    print("Starting speech-to-text transcription...")
    print("Press Ctrl+C to stop")
    transcribe_speech()