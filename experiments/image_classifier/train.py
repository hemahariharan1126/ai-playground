import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import mlflow
from torch.utils.tensorboard import SummaryWriter

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.logger import logger
from experiments.image_classifier.model import get_model

def train_model(epochs=5, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_experiment("Image_Classifier_CIFAR10")
    
    # Set up TensorBoard
    writer = SummaryWriter('logs/image_classifier')

    # 1. Data Preprocessing & Augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Collect Dataset (CIFAR-10)
    logger.info("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Build Model
    model = get_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.base_model.classifier.parameters(), lr=learning_rate)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("model_architecture", "MobileNetV2")

        # 4. Train Model
        logger.info("Starting training...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # 5. Evaluate Accuracy
        logger.info("Evaluating model...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        final_accuracy = 100 * correct / total
        logger.info(f"Test Accuracy: {final_accuracy:.2f}%")
        mlflow.log_metric("test_accuracy", final_accuracy)

        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/image_classifier_cifar10.pth'
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved and logged to MLflow artifacts.")

    writer.close()

if __name__ == "__main__":
    # For a quick dry run, use 1 epoch
    train_model(epochs=1)
