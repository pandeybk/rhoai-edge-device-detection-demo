# prediction_app/image_utils.py
from torchvision import transforms
from PIL import Image
import numpy as np

def prepare_image(image_path):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image)
    
    # Convert the transformed image to a list
    image_data = transformed_image.numpy().flatten().tolist()

    if len(image_data) != 3*224*224:
        raise ValueError("The number of elements in 'image_data' does not match the expected input size.")
    
    return image_data

def create_payload(image_data):
    # Construct the JSON payload
    payload = {
        "inputs": [{
            "name": "input",
            "shape": [1, 3, 224, 224],
            "datatype": "FP32",
            "data": image_data
        }]
    }
    return payload
