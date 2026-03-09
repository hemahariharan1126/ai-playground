# Experiment: Image Classifier

## 🤖 Model
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Framework**: PyTorch

## 📊 Dataset
- **Name**: CIFAR-10
- **Classes**: 10 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)

## 🛠️ How it works
We use a pre-trained MobileNetV2 backbone (trained on ImageNet) and replace the final classification head with a custom linear layer mapping to our 10 target classes. The backbone layers are frozen, ensuring rapid training and minimal compute requirements.

## 🚀 Run locally
### 1. Train the model
```bash
python experiments/image_classifier/train.py
```
This will automatically download the CIFAR-10 dataset, log results to MLflow/TensorBoard, and save the weights in `models/`.

### 2. Predict
```bash
python experiments/image_classifier/predict.py path/to/image.jpg
```
