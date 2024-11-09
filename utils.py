import torch
from PIL import Image
from torchvision import transforms

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing transform
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
    ])
    return transform

# Function to load the model from the saved file
def load_model(file_path):
    model = torch.load(file_path)  # Load the entire model
    model.to(device)  # Move to GPU or CPU
    model.eval()  # Set to evaluation mode
    return model

# Function for making predictions
def predict_image(image_path, model, transform):
    # Load and preprocess the image
    # image = Image.open(image_path)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

    with torch.no_grad():
        outputs = model(image).logits
        _, predicted = torch.max(outputs, 1)
    
    class_names = [
        'Ajwain Leaf', 'Almond Leaf', 'Ashoka Leaf', 'Bamboo Leaf', 'Banana Leaf', 
        'Coriander Leaf', 'Drumstick Leaf', 'Dumbcane Leaf', 'Eucalyptus Leaf', 
        'Fittonia Leaf', 'Gotu Kola Leaf', 'Maple Leaf', 'Guava Leaf', 
        'Hebiscus Leaf', 'Jackfruit Leaf', 'Lemon Leaf', 'Mango Leaf', 
        'Marigold Leaf', 'Mint Leaf', 'Neem Leaf', 'Papaya Leaf', 'Parijat Leaf', 
        'Peepal Leaf', 'Pine Leaf', 'Rose Leaf', 'Sagwan Leaf', 'Snake Plant Leaf', 
        'Tamarind Leaf', 'Tapioca Leaf', 'Tulsi Leaf'
        ]  # List of class names in the same order as your model outputs
    predicted_class_name = class_names[predicted.item()]

    # return predicted.item()  # Return the predicted class index
    return outputs, predicted_class_name

# Define the class names manually if they are known
# class_names = ['Aelovera', 'Bamboo', 'Banana', 'Coriander', 'Eucalyptus', 'Fittonia', 'Guava', 'Hebiscus', 'Jackfruit', 'Mango', 'Mint', 'Neem', 'Papaya', 'Peepal', 'Pine', 'Rose', 'Sagwan', 'Snake Plant', 'Tamarind', 'Tulsi']  # Replace with actual tree names

# Example of using the prediction function
# Load the model
# loaded_model = load_model_from_pickle('vit_model.pth')