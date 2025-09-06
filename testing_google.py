from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

synthesis_input = texttospeech.SynthesisInput(
    text="Hi! This is Google's Studio voice O. I sound a lot more natural than Wavenet. I am a text-to-speech voice created by Google. How are you today?" 
)

voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Chirp3-HD-Aoede"
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

with open("studio_output.mp3", "wb") as out:
    out.write(response.audio_content)
    print("✅ Audio saved as studio_output.mp3")
