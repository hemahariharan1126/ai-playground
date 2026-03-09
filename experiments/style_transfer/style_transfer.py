import torch
import torch.optim as optim
import os
import sys

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.style_transfer.model import get_model
from experiments.style_transfer.utils import load_image, save_image
from utils.logger import logger

def gram_matrix(tensor):
    """Calculate the Gram Matrix of a given tensor."""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def run_style_transfer(content_path, style_path, output_path, steps=500, content_weight=1, style_weight=1e6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load images
    content = load_image(content_path).to(device)
    # Style image must match content image shape
    style = load_image(style_path, shape=content.shape[-2:]).to(device)
    
    # 2. Build Model
    model = get_model().to(device)
    
    # 3. Get features
    content_features = model(content)
    style_features = model(style)
    
    # 4. Gram matrices for style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # 5. Initialize generated image (target) as a copy of content
    target = content.clone().requires_grad_(True).to(device)
    
    # 6. Weights for style layers
    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.2
    }
    
    # 7. Optimizer
    optimizer = optim.Adam([target], lr=0.003)
    
    # Set up MLflow
    mlflow.set_experiment("Style_Transfer")
    
    # Set up TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('logs/style_transfer')
    
    logger.info("Starting Style Transfer optimization...")
    with mlflow.start_run():
        mlflow.log_params({
            "steps": steps,
            "content_weight": content_weight,
            "style_weight": style_weight,
            "optimizer": "Adam"
        })
        
        for i in range(1, steps + 1):
            target_features = model(target)
            
            # Content Loss
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
            
            # Style Loss
            style_loss = 0
            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                _, d, h, w = target_feature.shape
                style_loss += layer_style_loss / (d * h * w)
                
            # Total Loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Log metrics
            if i % 10 == 0:
                mlflow.log_metric("total_loss", total_loss.item(), step=i)
                mlflow.log_metric("content_loss", content_loss.item(), step=i)
                mlflow.log_metric("style_loss", style_loss.item(), step=i)
                writer.add_scalar('Loss/total', total_loss.item(), i)
            
            if i % 100 == 0:
                logger.info(f"Step {i}: Total Loss: {total_loss.item():.4f}")

    writer.close()

    # 8. Save Result
    save_image(target, output_path)
    logger.info(f"Stylized image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        content = sys.argv[1]
        style = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else "stylized_output.png"
        run_style_transfer(content, style, out)
    else:
        logger.info("Usage: python style_transfer.py <content_image> <style_image> [output_image]")
