import numpy as np
import matplotlib.pyplot as plt
import logging

# Function to create confusion matrix
def create_confusion_matrix(true, pred):
    # Create empty confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for t, p in zip(true, pred):
        confusion_matrix[t][p] += 1
    return confusion_matrix

# Function to calculate metrics: accuracy, precision, recall, and f1
def calculate_metrics(confusion_matrix):

    num_classes = confusion_matrix.shape[0]
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Calculate global accuracy first
    total_samples = np.sum(confusion_matrix)
    correct_predictions = np.sum(np.diag(confusion_matrix))
    metrics['accuracy'] = correct_predictions / total_samples
    
    # Calculate metrics for each class
    for i in range(num_classes):
        # True Positives: diagonal elements
        tp = confusion_matrix[i][i]
        # False Positives: sum of column i excluding true positive
        fp = np.sum(confusion_matrix[:, i]) - tp
        # False Negatives: sum of row i excluding true positive
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
    
    # Calculate average metrics
    metrics['avg_precision'] = np.mean(metrics['precision'])
    metrics['avg_recall'] = np.mean(metrics['recall'])
    metrics['avg_f1'] = np.mean(metrics['f1'])
    
    return metrics

# Function to plot confusion matrix on a graph
def plot_confusion_matrix(ax, confusion_matrix, classes):
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmax=100)
    plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

# Function to plot training loss and test accuracy over epochs for MLP and CNN
def plot_training_metrics(train_losses, test_accuracies, model_type, epoch_interval=1):
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title(f'{model_type} Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot test accuracy
    plt.subplot(1, 2, 2)
    if epoch_interval > 1:
        # For MLP: Create proper epoch numbers
        epochs = range(0, len(train_losses), epoch_interval)
        plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    else:
        # For CNN: Use all epochs
        plt.plot(test_accuracies, 'r-', label='Test Accuracy')
    
    plt.title(f'{model_type} Test Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()