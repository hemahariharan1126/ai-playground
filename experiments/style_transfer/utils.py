import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path, max_size=400, shape=None):
    """Load and transform an image, ensuring it's <= max_size."""
    image = Image.open(image_path).convert('RGB')
    
    if max_size:
        size = max_size if max(image.size) > max_size else max(image.size)
    else:
        size = max(image.size)
        
    if shape:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    # Discard alpha channel (if any) and add batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    """Display a tensor as an image."""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def save_image(tensor, path):
    """Save a tensor as an image file."""
    image = im_convert(tensor)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)
