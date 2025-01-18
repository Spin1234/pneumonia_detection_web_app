from math import e
from flask import Flask, flash, request, jsonify
from flask_cors import CORS
import keras
import tensorflow as tf
import numpy as np
from PIL import Image


app = Flask(__name__)
CORS(app)


CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

model = tf.saved_model.load(r"Models/model_cnn_updated/2")


def process_image(image):

    img = Image.open(image).convert("RGB")
    image = img.resize((224,224))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # if img_array.shape[-1] == 1:
    #     # Convert grayscale to RGB by duplicating the single channel to 3 channels
    #     img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array))
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0

    return img_array

@app.route('/')
def home():
    return "Flask Server is Running. Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.files)
        input_data = request.files['file']

        if not input_data:
            return jsonify({'error': 'no file uploaded!'}), 400
        
        processed_image = process_image(input_data)
        print(f"Processed image shape: {processed_image.shape}")

        predictions = model(processed_image)

        return jsonify({'Prediction': CLASS_NAMES[np.argmax(predictions[0])]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

