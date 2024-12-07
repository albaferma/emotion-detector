
from flask import Flask, request, render_template_string
import cv2
from fer import FER
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

def detect_faces_and_emotions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta o formato del archivo.")
    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(img)
    faces = []
    emotion_texts = []
    for emotion in emotions:
        box = emotion["box"]
        faces.append(box)
        dominant_emotion = max(emotion["emotions"], key=emotion["emotions"].get)
        emotion_texts.append(dominant_emotion)
    return img, faces, emotion_texts

@app.route('/')
def upload_file():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detector de Emociones</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #d0e7f9; }
            .container { margin: auto; width: 50%; padding: 20px; }
            h1 { color: #00509e; }
            input[type="file"] { margin: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Detector de Emociones</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Detectar Emoción</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    try:
        _, faces, emotion_texts = detect_faces_and_emotions(filename)
        if not faces:
            result = "No se detectaron rostros en la imagen."
        else:
            result = "Emociones detectadas:<br>"
            for idx, emotion in enumerate(emotion_texts):
                result += f"Rostro {idx + 1}: {emotion}<br>"
    except Exception as e:
        result = f"Error: {str(e)}"
    os.remove(filename)
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resultado de Emoción</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; background-color: #d0e7f9; }}
            h1 {{ color: #00509e; }}
            p {{ font-size: 20px; }}
        </style>
    </head>
    <body>
        <h1>Resultado</h1>
        <p>{result}</p>
        <a href="/">Volver</a>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
