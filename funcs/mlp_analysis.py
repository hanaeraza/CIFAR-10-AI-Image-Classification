import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
from funcs.mlp import train_mlp, predict_mlp
from funcs.evaluation import calculate_metrics, create_confusion_matrix

logger = logging.getLogger('CIFAR10')

# MLP with variable number of hidden layers
class FlexibleMLP(nn.Module):
    def __init__(self, num_layers):
        super(FlexibleMLP, self).__init__()
        
        layers = []
        
        # Input layer: 50 -> 512
        layers.extend([
            nn.Linear(50, 512),
            nn.ReLU()
        ])
        
        # Hidden layers: 512 -> 512 with BatchNorm
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            ])
        
        # Output layer: 512 -> 10
        layers.append(nn.Linear(512, 10))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Class to analyze how MLP depth affects model performance
class MLPAnalyzer:
    def __init__(self, train_features, train_labels, test_features, test_labels, device):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.device = device
        self.results = None
        self.depths = None
    
    def run_analysis(self, depths=None):
        # Default depths if none provided
        if depths is None:
            depths = [1, 2, 3, 4, 5, 6]
        
        self.depths = depths
        logger.info(f"Starting MLP depth analysis on {len(depths)} models...")
        
        # Initialize results dictionary
        self.results = {
            'train_accuracy': [],
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': []
        }
        
        # Test each depth
        for depth in depths:
            # Create model using FlexibleMLP
            model = FlexibleMLP(depth).to(self.device)
            
            # Train model with verbose=False to prevent duplicate output
            trained_model, _, _ = train_mlp(
                self.train_features,
                self.train_labels,
                self.test_features,
                self.test_labels,
                self.device,
                model=model
            )
            
            # Get predictions
            train_pred = predict_mlp(trained_model, self.train_features, self.device)
            test_pred = predict_mlp(trained_model, self.test_features, self.device)
            
            # Calculate metrics and store results as before...
            train_conf_matrix = create_confusion_matrix(self.train_labels, train_pred)
            test_conf_matrix = create_confusion_matrix(self.test_labels, test_pred)
            
            train_metrics = calculate_metrics(train_conf_matrix)
            test_metrics = calculate_metrics(test_conf_matrix)
            
            # Store results
            self.results['train_accuracy'].append(train_metrics['accuracy'])
            self.results['test_accuracy'].append(test_metrics['accuracy'])
            self.results['test_precision'].append(test_metrics['avg_precision'])
            self.results['test_recall'].append(test_metrics['avg_recall'])
            self.results['test_f1'].append(test_metrics['avg_f1'])
            
            # Log progress through logger instead
            logger.info(f'Depth {depth}: Loss: {train_metrics["accuracy"]:.4f}, Test Accuracy: {test_metrics["accuracy"]:.4f}')
        
        logger.info("Depth analysis complete!")
    
    # Function to create and save visualizations of the analysis results.
    def plot_results(self, save_dir=None):
        if self.results is None:
            raise ValueError("No results to plot. Run analysis first.")
        
        plt.figure(figsize=(12, 6))
        plt.suptitle('MLP Depth Analysis', fontsize=16, y=0.98)
        plt.subplots_adjust(hspace=0.5, wspace=0, left=0.1, right=0.9, top=0.90, bottom=0.05)
        
        # Plot training vs testing accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.depths, self.results['train_accuracy'], 'b-', label='Training Accuracy')
        plt.plot(self.depths, self.results['test_accuracy'], 'r-', label='Testing Accuracy')
        plt.xlabel('Number of Hidden Layers')
        plt.ylabel('Accuracy')
        plt.title('Training vs Testing Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot test metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.depths, self.results['test_precision'], 'g-', label='Precision')
        plt.plot(self.depths, self.results['test_recall'], 'b-', label='Recall')
        plt.plot(self.depths, self.results['test_f1'], 'r-', label='F1 Score')
        plt.xlabel('Number of Hidden Layers')
        plt.ylabel('Score')
        plt.title('Test Set Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'mlp_depth_analysis.png'), 
                       bbox_inches='tight', dpi=300)
            logger.info(f"Saved depth analysis plot to {save_dir}")
        
        plt.show(block=False)
    
    # Function to display and optionally save a table of metrics for each network depth.
    def display_metrics(self, save_dir=None):
        if self.results is None:
            raise ValueError("No results to display. Run analysis first.")
        
        # Prepare table strings
        header = "MLP Depth Analysis:"
        divider = "-" * 80
        column_headers = (f"{'Layers':<8} {'Train Acc':>10} {'Test Acc':>10} "
                         f"{'Precision':>10} {'Recall':>10} {'F1':>10}")
        
        # Build table rows
        table_rows = []
        for i, depth in enumerate(self.depths):
            row = (f"{depth:<8} "
                   f"{self.results['train_accuracy'][i]:>10.3f} "
                   f"{self.results['test_accuracy'][i]:>10.3f} "
                   f"{self.results['test_precision'][i]:>10.3f} "
                   f"{self.results['test_recall'][i]:>10.3f} "
                   f"{self.results['test_f1'][i]:>10.3f}")
            table_rows.append(row)
        
        # Create complete table string
        table_content = "\n".join([
            header,
            divider,
            column_headers,
            divider,
            "\n".join(table_rows),
            divider
        ])
        
        # Log to console
        logger.info("\n" + table_content)
        
        # Save if directory provided
        if save_dir:
            save_path = os.path.join(save_dir, 'mlp_depth_metrics.txt')
            with open(save_path, 'w') as f:
                f.write(table_content)
            logger.info(f"Saved depth analysis metrics to {save_path}")