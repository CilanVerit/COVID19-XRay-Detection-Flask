import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('concatenate-fold3.hdf5')

# Define class labels
CLASS_LABELS = ['Covid_19', 'Normal', 'Pneumonia']

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define the folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resizing image to match model input size
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image and get prediction
        image = preprocess_image(filepath)
        prediction = model.predict(image)
        class_idx = np.argmax(prediction)
        predicted_label = CLASS_LABELS[class_idx]

        # Get prediction probabilities for each class
        prediction_probabilities = {CLASS_LABELS[i]: round(float(prediction[0][i]), 4) for i in range(len(CLASS_LABELS))}

        # Return the result to the user
        return render_template('result.html', label=predicted_label, filename=filename, probabilities=prediction_probabilities)

    return redirect(request.url)

# Route to display the uploaded image
@app.route('/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
