import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class VGG11(nn.Module):
    
    def __init__(self):
        super(VGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: Input RGB image (3 channels) -> 64 feature maps
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 2: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 128 -> 256 -> 256 feature maps (two conv layers)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 256 -> 512 -> 512 feature maps (two conv layers)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 512 -> 512 -> 512 feature maps (two conv layers)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            # First FC layer: Flattened features -> 4096
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            
            # Second FC layer: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Output layer: 4096 -> 10 (
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.features(x)  
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x

# Function to train CNN using VGG11 model
def train_cnn(train_loader, test_loader, device, model=None, num_epochs=50):
    # Initialize model and move to specified device
    if model is None:
        model = VGG11().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Learning rate scheduler to reduce LR when accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    # Track best model and metrics
    best_accuracy = 0.0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train() 
        running_loss = 0.0
        
        # Iterate over training batches with progress bar
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        # Update learning rate based on validation accuracy
        scheduler.step(accuracy)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f'New best model! Accuracy: {accuracy:.2f}%')
        
        print(f'Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Test Accuracy = {accuracy:.2f}%')
    
    return model, train_losses, test_accuracies

# Function to predict using trained VGG11 model
def predict_cnn(model, test_loader, device):
    model.eval() 
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    return np.array(all_predictions)