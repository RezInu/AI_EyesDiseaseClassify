import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained model
model_path = "model/AI_BCNN_model.h5"
model = load_model(model_path)

# Define the target size for the input images
target_size = (227, 227)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        classes = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']  # Add your class labels here
        predicted_class = classes[np.argmax(prediction)]

        return render_template('index.html', message='Prediction: {}'.format(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)