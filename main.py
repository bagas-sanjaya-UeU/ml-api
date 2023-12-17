from flask import Flask, Response, render_template, request, redirect, url_for, flash, jsonify
from google.cloud import storage
import os
# import tensorflow as tf
# from io import BytesIO
# from keras.models import load_model
# import numpy as np
# import soundfile as sf
# import librosa


app = Flask(__name__)

# # storage_client = storage.Client(project='', credentials='credentials.json')
# # bucket = storage_client.list_buckets()
# def CTCLoss(y_true, y_pred):
#     # Compute the training-time loss value
#     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

#     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

#     loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
#     return loss

# loaded_model = load_model('my_model.keras', custom_objects={'CTCLoss': CTCLoss})

# # An integer scalar Tensor. The window length in samples.
# frame_length = 256
# # An integer scalar Tensor. The number of samples to step.
# frame_step = 160
# # An integer scalar Tensor. The size of the FFT to apply.
# # If not provided, uses the smallest power of 2 enclosing frame_length.
# fft_length = 384

# def load_and_preprocess_audio(audio_file):
#     # 1. Read wav file
#     audio, sr = librosa.load(audio_file , sr=None)
    
#     # 2. Resample the audio to 22050 Hz if it's not already
#     if sr != 22050:
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    
#     # 3. Convert float audio to int16
#     audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    
#     # 4. Save the audio file in 16-bit PCM WAV format
#     sf.write('resampled_audio.wav', audio, 22050, subtype='PCM_16')
    
#     # 5. Read the resampled audio file
#     audio, _ = librosa.load('resampled_audio.wav', sr=None)
    
#     # 6. Change type to float
#     audio = tf.cast(audio, tf.float32)
    
#     # 7. Get the spectrogram
#     spectrogram = tf.signal.stft(
#         audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
#     )
    
#     # 8. We only need the magnitude, which can be derived by applying tf.abs
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.math.pow(spectrogram, 0.5)
    
#     # 9. Normalisation
#     means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
#     stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
#     spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
#     return spectrogram

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try: 
            url = request.form['url']
            # debug if url is not empty
            return Response(url, status=200)
        except:
            return Response('gagal', status=400)

    return render_template('index.html')   
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 


