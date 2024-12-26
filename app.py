from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import numpy as np
import cv2
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('handwritten_equation_model.h5')

# Define the label map with indices
label_map = {
    0: 'equation1',
    1: 'equation2',
    2: 'equation3',
    3: 'equation4',
    4: 'equation5',
    5: 'equation6',
    6: 'equation7',
    7: 'equation8',
    8: 'equation9',
    9: 'equation10',
}

# Define the LaTeX map
latex_map = {
    'equation1': 'x^2',
    'equation2': '\\sqrt{x}',
    'equation3': '\\sqrt[3]{x}',
    'equation4': '\\frac{x}{y}',
    'equation5': '\\frac{1}{2}',
    'equation6': 'ax+b=0',
    'equation7': 'ax^2+bx+c=0',
    'equation8': '\\delta=b^2-4ac',
    'equation9': '(ab)^n=a^nb^n',
    'equation10': '(a^m)^n=a^{mn}',
}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_image(image_data):
    # Decode the image
    image = base64.b64decode(image_data)
    np_image = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    img = cv2.resize(img, (28, 28))  # Resize to the input size of your model
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json()
    image_data = data['image']

    # Preprocess the image
    processed_image = preprocess_image(image_data)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability

    # Map the index to the class name
    predicted_class_name = label_map.get(predicted_class_index, 'Unknown equation')

    # Get the LaTeX string using the class name
    predicted_latex = latex_map.get(predicted_class_name, 'Unknown equation')

    return jsonify({'latex': predicted_latex})

if __name__ == '__main__':
    app.run(debug=True, port=5000)