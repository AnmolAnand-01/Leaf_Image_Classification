from flask import Flask, render_template, request, redirect
import os
from utils import load_model, predict_image, get_transform

app = Flask(__name__)

# Path to your model
MODEL_PATH = os.path.join('model', 'vit_model_08_11_2024.pth')

# Load the model
model = load_model(MODEL_PATH)
transform = get_transform()

# Home route with collage display
@app.route('/')
def index():
    # classes = ['Ajwain', 'Almond', 'Ashoka', 'Bamboo', 'Banana', 'Coriander', 'Drumstick', 'Dumbcane', 'Eucalyptus', 'Fittonia', 'Gotu kola', 'Maple', 'Guava', 'Hebiscus', 'Jackfruit', 'Lemon', 'Mango', 'Marigold', 'Mint', 'Neem', 'Papaya', 'Parijat', 'Peepal', 'Pine', 'Rose', 'Sagwan', 'Snake Plant', 'Tamarind', 'Tapioca', 'Tulsi']
    classes = [
    'Ajwain Leaf', 'Almond Leaf', 'Ashoka Leaf', 'Bamboo Leaf', 'Banana Leaf', 
    'Coriander Leaf', 'Drumstick Leaf', 'Dumbcane Leaf', 'Eucalyptus Leaf', 
    'Fittonia Leaf', 'Gotu Kola Leaf', 'Maple Leaf', 'Guava Leaf', 
    'Hebiscus Leaf', 'Jackfruit Leaf', 'Lemon Leaf', 'Mango Leaf', 
    'Marigold Leaf', 'Mint Leaf', 'Neem Leaf', 'Papaya Leaf', 'Parijat Leaf', 
    'Peepal Leaf', 'Pine Leaf', 'Rose Leaf', 'Sagwan Leaf', 'Snake Plant Leaf', 
    'Tamarind Leaf', 'Tapioca Leaf', 'Tulsi Leaf'
    ]


    # Get all image files in the static/images directory
    image_folder = os.path.join(app.static_folder, 'images')
    image_files = os.listdir(image_folder)

    print("Image files in static/images folder:", image_files)
    return render_template('index.html', classes=classes, image_files=image_files)

# Image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image_path = os.path.join('static/images', file.filename)
        file.save(image_path)
        
        # Predict the class of the image
        prediction, predicted_class_name = predict_image(image_path, model, transform)
        # print(f'Saving image to: {image_path}')  # Add this line for debugging
        return render_template('result.html', prediction=prediction, predicted_class_name=predicted_class_name, image_path=image_path)

