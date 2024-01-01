import os
from flask import Flask, render_template, request
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')

# Define the custom layer (MCDropout) before loading the model
class MCDropout(tensorflow.keras.layers.Dropout):
    def call(self, inputs):
     return super().call(inputs, training = True)

# Load the model with custom layers
with tensorflow.keras.utils.custom_object_scope({'MCDropout': MCDropout}):
    model = load_model('model/AI_BCNN_model.h5')

# Define the target size for the input images
target_size = (227, 227)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join('static/uploads', filename) 
            file.save(file_path)

            # Preprocess the image for prediction
            img_array = preprocess_image(file_path)

            # Make prediction using the loaded model
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            # Map the predicted class to a human-readable label (modify as per your model)
            labels = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']
            predicted_label = labels[predicted_class]

            return render_template('index.html', image_path=f'uploads/{filename}', prediction=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)