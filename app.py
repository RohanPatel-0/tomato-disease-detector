from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import os
import io
import base64

app = Flask(__name__)
class_names = ["Bacterial Spot", "Early Blight", "Healthy", "Late Blight", 
               "Leaf Mold", "Powdery Mildew", "Septoria Leaf Spot", "Spider Mites Two-spotted Spider Mite",
               "Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus"]

model = load_model("model-creation/saved_model.keras")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256)) 
    image = img_to_array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    predicted_class_label = None
    if request.method == 'POST':
        try:
            image = Image.open(request.files['image'].stream)
            image = preprocess_image(image)
            pred = model.predict(image)
            predicted_class_index = np.argmax(pred)
            predicted_class_label = class_names[predicted_class_index]
        except Exception as err:
            print(f"Error processing image: {err}")
    return render_template('home.html', prediction=predicted_class_label)

if __name__ == '__main__':
    app.run(debug=True)
