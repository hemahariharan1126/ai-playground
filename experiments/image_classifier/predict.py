import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.image_classifier.model import get_model
from utils.logger import logger

def predict(image_path, model_path='models/image_classifier_cifar10.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR-10 Classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load Model
    model = get_model(num_classes=10)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded weights from {model_path}")
    else:
        logger.warning(f"No weights found at {model_path}. Using uninitialized model.")
    
    model.to(device)
    model.eval()
    
    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = outputs.max(1)
            
        class_name = classes[predicted.item()]
        logger.info(f"Prediction for {image_path}: {class_name}")
        return class_name
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        logger.info("Usage: python predict.py <path_to_image>")
