import torch
import torch.nn as nn
from torchvision import models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace the classifier head
        # MobileNetV2's classifier input features is 1280
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def get_model(num_classes=10):
    return ImageClassifier(num_classes=num_classes)
