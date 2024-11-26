import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger('CIFAR10')

class CIFAR10MLP(nn.Module):

    # Function to initialize MLP with flexible number of layers and sizes
    def __init__(self):
        super(CIFAR10MLP, self).__init__()
        self.model = nn.Sequential(
            # First layer: input(50) -> hidden(512)
            nn.Linear(50, 512),
            nn.ReLU(),
            
            # Second layer: hidden(512) -> hidden(512) with batch normalization
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            # Output layer: hidden(512) -> output(10)
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.model(x)

# Function train MLP 
def train_mlp(train_features, train_labels, test_features, test_labels, device, model=None, num_epochs=100):
    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_test = torch.FloatTensor(test_features).to(device)
    y_test = torch.LongTensor(test_labels).to(device)
    
    # Initialize model if not provided
    if model is None:
        model = CIFAR10MLP().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test).sum().item() / len(y_test)
                test_accuracies.append(accuracy)
                
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
    
    return model, train_losses, test_accuracies

# Function to predict using trained MLP model
def predict_mlp(model, features, device):
    model.eval()
    X = torch.FloatTensor(features).to(device)
    
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.cpu().numpy()