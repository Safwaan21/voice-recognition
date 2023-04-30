import librosa
import pydub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def convert_m4a_to_wav(m4a_path, wav_path):
    # Load the .m4a file
    audio = pydub.AudioSegment.from_file(m4a_path, format="m4a")
    # Export the audio as .wav file
    audio.export(wav_path, format="wav")


voices = ['fahtema', 'mom', 'saf']

def generate_spectrograms(category):
    audioPath = './data/usable/' + category + '.wav'
    newPath = './dataset/' + category + '/'
    
    amplTimeDom, sr = librosa.load(audioPath)
    n_fft = 2048
    hop_length = 1024
    stft_list = []
    for i in range(0, len(amplTimeDom) - n_fft, hop_length):
        segment = amplTimeDom[i:i + n_fft]  # Extract a segment of audio
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
        stft_list.append(stft)
    for i in range(0, len(stft_list)):

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(abs(stft_list[i])), sr=sr, hop_length=hop_length)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig( (newPath + category + str(i) + '.png'))
        plt.close()
        image = Image.open(newPath + category + str(i) + '.png')
        image = image.resize((200, 80))
        image.save(newPath + category + str(i) + '.png')

for voice in voices:
    generate_spectrograms(voice)





