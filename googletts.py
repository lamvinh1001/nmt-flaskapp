import wikipedia
import playsound
import os
from gtts import gTTS

wikipedia.set_lang('vi')
language = 'vi'


def speak(text):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", True)
    os.remove("sound.mp3")
