import os
import threading
import time
import whisper
import pyaudio
import numpy as np
from flask import Flask
from collections import deque
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model
model = whisper.load_model('base')

# Configure the Google Generative AI with your API key
genai.configure(api_key="AIzaSyCXWicT0jLpv6DtnHzj9lO21zsrxuc2Fcw")

# Configure the generative model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model_gen = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Parameters
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 2  # seconds
silence_buffer = deque(maxlen=int(RATE / CHUNK * SILENCE_THRESHOLD))

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize a variable to store partial transcripts
partial_transcript = ""

def recognize_speech():
    global partial_transcript
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    audio_buffer = []
    silence_count = 0

    while True:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

        # Convert audio data to numpy array for silence detection
        audio_data = np.frombuffer(data, dtype=np.int16)
        silence_buffer.append(np.abs(audio_data).mean())

        if np.mean(silence_buffer) < 100:  # Adjust threshold as needed
            silence_count += 1
        else:
            silence_count = 0

        if silence_count > int(RATE / CHUNK * SILENCE_THRESHOLD):
            if audio_buffer:
                # Save the audio data to a temporary file
                audio_data = b''.join(audio_buffer)
                audio_buffer = []  # Reset buffer
                silence_count = 0  # Reset silence count

                # Transcribe using Whisper
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_data)
                result = model.transcribe("temp_audio.wav", language='en')
                transcription = result.get('text', '').strip()

                if transcription:
                    print(f"You said: {transcription}")
                    if "hey adam" in transcription.lower():
                        command = transcription.lower().split("hey adam")[1].strip()
                        send_to_gemini(command)
                        partial_transcript = ""  # Reset after sending command
                    else:
                        partial_transcript += " " + transcription
                os.remove("temp_audio.wav")  # Clean up temporary file

def send_to_gemini(command):
    try:
        print(f"Sending command to Gemini: {command}")
        chat_session = model_gen.start_chat(history=[])
        response = chat_session.send_message(command)
        print("Gemini response:", response.text)
    except Exception as e:
        print(f"Error sending to Gemini: {e}")

@app.route('/')
def home():
    return "Speech Trainer is running."

if __name__ == '__main__':
    listener_thread = threading.Thread(target=recognize_speech, daemon=True)
    listener_thread.start()
    app.run(port=5000)
