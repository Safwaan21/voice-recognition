import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow import keras
import pydub
from collections import defaultdict
import librosa
import random
import os

trainedModel = keras.models.load_model("mymodel")
class_names = ['saf']
batch_size = 32
img_height = 80
img_width = 200

def get_prediction(audio_file):
    def convert_m4a_to_wav(m4a_path, wav_path):
        # Load the .m4a file
        audio = pydub.AudioSegment.from_file(m4a_path, format="m4a")
        # Export the audio as .wav file
        audio.export(wav_path, format="wav")
    
    nameOfWav = str(random.randint(1, 100000))

    convert_m4a_to_wav(audio_file, './data/usable/' + nameOfWav + '.wav')

    def generate_spectrograms(category):
        audioPath = './data/usable/' + category + '.wav'
        os.makedirs('./dataset/' + category)
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
        return len(stft_list)

    num_images = generate_spectrograms(nameOfWav)

    def load_images(path):
        img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = trainedModel.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return[class_names[np.argmax(score)], 100 * np.max(score)]

    ans = defaultdict(int)
    for i in range(0, num_images):
        path = './dataset/' + nameOfWav + '/' + nameOfWav + str(i) + '.png'
        retVal = load_images(path=path)
        ans[retVal[0]] += 1
    print(ans)
