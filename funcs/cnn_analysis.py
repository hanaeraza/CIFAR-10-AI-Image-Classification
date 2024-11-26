import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import time
from funcs.cnn import train_cnn

logger = logging.getLogger('CIFAR10')

# A flexible VGG-style network that can vary in depth and kernel sizes.
# Can very depth up to 10 layers. 
class FlexibleVGG(nn.Module):
    def __init__(self, num_conv_blocks=8, kernel_size=3):
        super(FlexibleVGG, self).__init__()
        
        # Define channel sizes with one additional block beyond VGG11
        channel_blocks = [
            [64],       # Block 1: conv->pool
            [128],      # Block 2: conv->pool
            [256, 256], # Block 3: conv->conv->pool
            [512, 512], # Block 4: conv->conv->pool
            [512, 512], # Block 5: conv->conv->pool (Original VGG11 ends here)
            [512, 512]  # Block 6: Additional block
        ]
        
        # Take the first n blocks based on depth
        num_blocks = (num_conv_blocks + 2) // 2  # Convert depth to number of blocks
        channel_blocks = channel_blocks[:num_blocks]
        
        features = []
        in_channels = 3  # Input channels (RGB)
        
        # Create convolutional blocks
        for block in channel_blocks:
            for out_channels in block:
                features.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=1, padding=kernel_size//2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
                in_channels = out_channels
            
            # Add MaxPool after each block
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features = nn.Sequential(*features)

        # Add adaptive pooling to ensure consistent feature size and optimize training time for my machine
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier remains the same as VGG11
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Class for analyzing how CNN architecture choices affect model performance.
class CNNAnalyzer:
    def __init__(self, train_loader, test_loader, device):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.results_depth = None
        self.results_kernel = None
    
    # Function to analyze how network depth affects performance.
    def analyze_depth(self, depths=None, num_epochs=10):
        if depths is None:
            depths = [2, 4, 6, 8, 10]  # All possible depths up to extended VGG11
        
        logger.info("Starting CNN depth analysis...")
        self.results_depth = {
            'depths': depths,
            'accuracies': [],
            'training_times': []
        }
        
        for depth in depths:
            try:
                start_time = time.time()
                model = FlexibleVGG(num_conv_blocks=depth).to(self.device)
                
                # Use existing training function from cnn.py
                trained_model, losses, accuracies = train_cnn(
                    self.train_loader, 
                    self.test_loader, 
                    self.device,
                    model=model,
                    num_epochs=num_epochs
                )
                
                training_time = time.time() - start_time
                final_accuracy = accuracies[-1]
                
                self.results_depth['accuracies'].append(final_accuracy)
                self.results_depth['training_times'].append(training_time)
                
                logger.info(f"Depth {depth}: Accuracy: {final_accuracy:.2f}%, Time: {training_time:.2f}s")
            except Exception as e:
                logger.error(f"Error training CNN with depth {depth}: {str(e)}")
                self.results_depth['accuracies'].append(0)
                self.results_depth['training_times'].append(0)
        logger.info("Depth analysis complete!")
    
    # Function to analyze how kernel size affects performance.
    def analyze_kernel_sizes(self, kernel_sizes=None, num_epochs=10):
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 5, 7]  # Kernel sizes to test
        
        logger.info("Starting CNN kernel size analysis...")
        self.results_kernel = {
            'kernel_sizes': kernel_sizes,
            'accuracies': [],
            'training_times': []
        }
        
        for kernel_size in kernel_sizes:
            try:
                start_time = time.time()
                model = FlexibleVGG(kernel_size=kernel_size).to(self.device)
                
                # Use existing training function from cnn.py
                trained_model, losses, accuracies = train_cnn(
                    self.train_loader, 
                    self.test_loader, 
                    self.device,
                    model=model,
                    num_epochs=num_epochs
                )
                
                training_time = time.time() - start_time
                final_accuracy = accuracies[-1]
                
                self.results_kernel['accuracies'].append(final_accuracy)
                self.results_kernel['training_times'].append(training_time)
                
                logger.info(f"Kernel {kernel_size}x{kernel_size}: Accuracy: {final_accuracy:.2f}%, Time: {training_time:.2f}s")
            except Exception as e:
                logger.error(f"Error training CNN with kernel size {kernel_size}: {str(e)}")
                self.results_kernel['accuracies'].append(0)
                self.results_kernel['training_times'].append(0)
        logger.info("Kernel size analysis complete!")
    

    # Function to create visualizations of the analysis results.
    def plot_results(self, save_dir=None):
        # Plot depth analysis results
        if self.results_depth is not None:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.results_depth['depths'], 
                    self.results_depth['accuracies'], 'b-o', linewidth=2)
            plt.xlabel('Number of Convolutional Layers')
            plt.ylabel('Accuracy (%)')
            plt.title('CNN Depth vs Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.results_depth['depths'], 
                    self.results_depth['training_times'], 'r-o', linewidth=2)
            plt.xlabel('Number of Convolutional Layers')
            plt.ylabel('Training Time (seconds)')
            plt.title('CNN Depth vs Training Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'cnn_depth_analysis.png'), bbox_inches='tight', dpi=300)
                logger.info(f"Saved depth analysis plot to {save_dir}")
            plt.show(block=False)
        
        # Plot kernel size analysis results
        if self.results_kernel is not None:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.results_kernel['kernel_sizes'], 
                    self.results_kernel['accuracies'], 'b-o', linewidth=2)
            plt.xlabel('Kernel Size')
            plt.ylabel('Accuracy (%)')
            plt.title('Impact of Kernel Size on Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(self.results_kernel['kernel_sizes'], 
                      [f'{k}×{k}' for k in self.results_kernel['kernel_sizes']])
            
            plt.subplot(1, 2, 2)
            plt.plot(self.results_kernel['kernel_sizes'], 
                    self.results_kernel['training_times'], 'r-o', linewidth=2)
            plt.xlabel('Kernel Size')
            plt.ylabel('Training Time (seconds)')
            plt.title('Impact of Kernel Size on Computational Cost')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(self.results_kernel['kernel_sizes'], 
                      [f'{k}×{k}' for k in self.results_kernel['kernel_sizes']])
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'kernel_size_analysis.png'), bbox_inches='tight', dpi=300)
                logger.info(f"Saved kernel size analysis plot to {save_dir}")
            plt.show(block=False)
    
    # Function to display and save analysis metrics.
    def display_metrics(self, save_dir=None):
        # Common formatting values
        divider = "-" * 60
        accuracy_width = 14  # Width for accuracy column
        time_width = 11     # Width for time column
        
        # Display depth analysis metrics
        if self.results_depth is not None:
            header = "CNN Depth Analysis Results:"
            column_headers = (f"{'Depth':<11} "
                            f"{'Accuracy (%)':<{accuracy_width}} "
                            f"{'Time (s)':<{time_width}}")
            
            table_rows = []
            for i, depth in enumerate(self.results_depth['depths']):
                row = (f"{depth:<11} "
                    f"{self.results_depth['accuracies'][i]:>{accuracy_width}.2f} "
                    f"{self.results_depth['training_times'][i]:>{time_width}.2f}")
                table_rows.append(row)
            
            if len(self.results_depth['depths']) > 1:
                table_rows.append(divider)
                acc_change = self.results_depth['accuracies'][-1] - self.results_depth['accuracies'][0]
                time_change = self.results_depth['training_times'][-1] - self.results_depth['training_times'][0]
                
                summary = (f"{'Change:':<11} "
                        f"{acc_change:>{accuracy_width}.2f} "
                        f"{time_change:>{time_width}.2f}")
                table_rows.append(summary)
            
            depth_table = "\n".join([
                header, divider, column_headers, divider,
                "\n".join(table_rows), divider
            ])
            
            logger.info("\n" + depth_table)
            
            if save_dir:
                save_path = os.path.join(save_dir, 'cnn_depth_metrics.txt')
                with open(save_path, 'w') as f:
                    f.write(depth_table)
                logger.info(f"Saved depth analysis metrics to {save_path}")
        
        # Display kernel size analysis metrics
        if self.results_kernel is not None:
            header = "\nCNN Kernel Size Analysis Results:"
            column_headers = (f"{'Kernel Size':<11} "
                            f"{'Accuracy (%)':<{accuracy_width}} "
                            f"{'Time (s)':<{time_width}}")
            
            table_rows = []
            for i, size in enumerate(self.results_kernel['kernel_sizes']):
                row = (f"{size}x{size:<9} "
                    f"{self.results_kernel['accuracies'][i]:>{accuracy_width}.2f} "
                    f"{self.results_kernel['training_times'][i]:>{time_width}.2f}")
                table_rows.append(row)
            
            if len(self.results_kernel['kernel_sizes']) > 1:
                table_rows.append(divider)
                acc_change = self.results_kernel['accuracies'][-1] - self.results_kernel['accuracies'][0]
                time_change = self.results_kernel['training_times'][-1] - self.results_kernel['training_times'][0]
                
                summary = (f"{'Change:':<11} "
                        f"{acc_change:>{accuracy_width}.2f} "
                        f"{time_change:>{time_width}.2f}")
                table_rows.append(summary)
            
            kernel_table = "\n".join([
                header, divider, column_headers, divider,
                "\n".join(table_rows), divider
            ])
            
            logger.info("\n" + kernel_table)
            
            if save_dir:
                save_path = os.path.join(save_dir, 'kernel_size_metrics.txt')
                with open(save_path, 'w') as f:
                    f.write(kernel_table)
                logger.info(f"Saved kernel size analysis metrics to {save_path}")