import torch
import torchvision
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Function extracts features using pre-trained ResNet18 model
def extract_features(trainloader, device): 
    
    # Set model to use as ResNet18
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model = model.to(device)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove last layer
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(trainloader):
            images = images.to(device)
            features.append(model(images).cpu().numpy())  # Extract features
            labels.append(targets.numpy())  # Collect labels

    # Reshape arrays
    features = np.vstack(features)
    labels = np.hstack(labels)

    features = features.reshape(features.shape[0], -1)

    return features, labels
