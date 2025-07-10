from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("digit_recognition_model.h5")

# Function to preprocess uploaded image
def preprocess_image(image):
    import cv2
    import numpy as np
    
    # Read image as grayscale
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Invert colors if background is black (sometimes necessary)
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)

    # Resize to 28x28
    image = cv2.resize(image, (28, 28))

    # Normalize and reshape for the model
    image = image.astype("float32") / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Add batch & channel dimensions

    return image


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# API Route to predict digit
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    image = preprocess_image(file)
    
    prediction = model.predict(image)
    digit = int(np.argmax(prediction))  
    
    return jsonify({'prediction': digit})

if __name__ == '__main__':
    app.run(debug=True)
