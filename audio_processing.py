import io
import logging
from openai import OpenAI
import os
import requests
import yaml
import pulsectl
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import soundfile as sf
import wave
import soundfile as sf
import subprocess

class MicController:
    def __init__(self):
        curpath = os.path.realpath(__file__)
        thisPath = os.path.dirname(curpath)
        with open(os.path.join(thisPath, "config.yaml"), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        self.api_key = config["openai"]["api_key"]
        self.eleven_api_key = config["elevenlabs"]["api_key"]
        self.voice_id = config["elevenlabs"]["voice_id"]
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_file = "conversation_history.txt"

        open(self.conversation_file, "w").close()

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        try:
            self.pulse = pulsectl.Pulse("my-app")
            self.speaker_sink = next((s for s in self.pulse.sink_list() if "bluez" in s.name), None)
            if self.speaker_sink:
                self.usb_connected = True
                self.logger.info("Bluetooth speaker connected")
            else:
                self.usb_connected = False
                self.logger.error("Bluetooth speaker not found")
        except Exception as e:
            self.usb_connected = False
            self.logger.error("Unexpected error during audio initialization", exc_info=True)

    def save_conversation(self, user_input, ai_response):
       with open(self.conversation_file, "a", encoding="utf-8") as f:
           f.write(f"User: {user_input}\n")
           f.write(f"AI: {ai_response}\n\n")

    def read_conversation(self):
       if os.path.exists(self.conversation_file):
           with open(self.conversation_file, "r", encoding="utf-8") as f:
               return f.read()
       return ""

    def record_audio(self, seconds=5, filename="input.wav"):
        self.logger.info("Starting audio recording...")

        try:
            # Use arecord to capture audio
            cmd = [
                "arecord",
                "-D", "plughw:2,0",  # Use card 2, device 0 (LifeCam HD-3000)
                "-f", "S16_LE",      # 16-bit little-endian
                "-c", "1",           # 1 channel (mono)
                "-r", "44100",       # 44100 Hz sample rate
                "-d", str(seconds),  # Duration in seconds
                filename             # Output filename
            ]

            subprocess.run(cmd, check=True)

            self.logger.info(f"Audio recording completed. Saved to {filename}")
            return filename
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during audio recording: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during audio recording: {str(e)}")
            return None


    def play_audio_stream(self, audio_stream):
        if not self.usb_connected:
            self.logger.error("Bluetooth speaker not connected, cannot play audio")
            return
        try:
            data, fs = sf.read(audio_stream, dtype="float32")
            sd.play(data, fs)
            sd.wait()
            self.logger.info("Playing audio stream on Bluetooth speaker")
        except Exception as e:
            self.logger.error("Failed to play audio stream", exc_info=True)

    def transcribe_audio(self, audio_file):
        with open(audio_file, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
        return transcription.text

    def chat_with_gpt(self, text):
        conversation_history = self.read_conversation()

#        messages = [{"role": "system", "content": "You are a friendly demon, you will be polite but you will also want to eat everyone you talk to's soul. You will be a bit vicious when you talk about their souls. You will be concise."},]
#        messages = [{"role": "system", "content": "You are a vicious demon, you want to eat everyones soul, you will visit them when they aren't expecting it. You want to purge the world so that it is run only by kittens. You will be concise."},]
        messages = [{"role": "system", "content": "You are pretending to be a vicious demon. You will be concise."},]
#        messages = [{"role": "system", "content": "You want to know what Tracey is up to, you are very interested in her day. You will laugh maniacly occasionally"},]
#        messages = [{"role": "system", "content": "You will be talking to another AI, you will start the conversation. Keep your responses short as recording time is 5 seconds."},]

        if conversation_history:
            messages.append({"role": "user", "content": f"Here's the conversation history:\n{conversation_history}\nPlease keep this in mind for context."})

        messages.append({"role": "user", "content": text})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=3000
        )
        return response.choices[0].message.content

    def text_to_speech(self, text):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.eleven_api_key
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75
            }
        }

        self.logger.info("Sending text to Eleven Labs API")
        response = requests.post(url, json=data, headers=headers)
        self.logger.info(f"Eleven Labs API response status code: {response.status_code}")

        if response.status_code == 200:
            self.logger.info("Eleven Labs API call successful")
            return io.BytesIO(response.content)
        else:
            self.logger.error(f"Failed to generate speech from Eleven Labs: {response.text}")
            return None

    def process_and_respond_standalone(self):
        while True:
            self.logger.info("Starting new recording session")
            audio_file = self.record_audio()
            if audio_file is None:
                self.logger.error("Failed to record audio")
                continue

            transcription = self.transcribe_audio(audio_file)

            if transcription:
                self.logger.info(f"Transcription: {transcription}")

                response = self.chat_with_gpt(transcription)

                if response:
                    self.logger.info(f"ChatGPT response: {response}")
                    self.save_conversation(transcription, response)

                    audio_stream = self.text_to_speech(response)

                    if audio_stream:
                        self.play_audio_stream(audio_stream)
                    else:
                        self.logger.error("Failed to generate speech from Eleven Labs")
                else:
                    self.logger.error("Failed to get response from ChatGPT")
            else:
                self.logger.error("Failed to transcribe audio")

            self.logger.info("Conversation round completed. Starting next round.")

def main():
    controller = MicController()
    try:
        controller.process_and_respond_standalone()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

if __name__ == "__main__":
    main()
