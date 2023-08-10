import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow import keras
import librosa
import audioread
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = keras.models.load_model('final_model.h5')

emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad']

def preprocess_audio(audio_path):
    if not os.path.exists(audio_path):
        return None
    try:
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            audio = f.read(dtype='float32')
        audio /= np.max(np.abs(audio))
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        if mfcc.shape[1] > 100:
            mfcc = mfcc[:, :100]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        return mfcc
    except Exception as e:
        print(f"Error al procesar el archivo de audio: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió ningún archivo de audio.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo inválido.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido. Solo se permiten archivos WAV.'}), 400

    audio_path = 'temp_audio.wav'
    try:
        file.save(audio_path)
    except Exception as e:
        return jsonify({'error': f'Error al guardar el archivo de audio: {e}'}), 500

    mfcc = preprocess_audio(audio_path)
    if mfcc is None:
        return jsonify({'error': 'Error al procesar el archivo de audio. Asegúrate de que sea un archivo de audio válido.'}), 500

    predictions = model.predict(mfcc)
    emotion_index = np.argmax(predictions)
    emotion = emotions[emotion_index]
    probability = predictions[0][emotion_index]

    gif_url = get_gif_url(emotion)

    return jsonify({'emotion': emotion, 'probability': float(probability), 'gif_url': gif_url}), 200

def get_gif_url(emotion):
    gif_folder = "static/emojis"
    if emotion == "Felicidad":
        return f"{gif_folder}/felizAndres.gif"
    elif emotion == "Ansiedad":
        return f"{gif_folder}/ansiedadAndres.gif"
    elif emotion == "Curiosidad":
        return f"{gif_folder}/curiosidadAndres.gif"
    elif emotion == "Tranquilidad":
        return f"{gif_folder}/tranquilidadAndres.gif"
    else:
        return f"{gif_folder}/tranquilidadAndres.gif"
    
if __name__ == '__main__':
    app.run(debug=True)
