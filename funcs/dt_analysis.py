import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
from funcs.decision_tree import train_decision_tree, predict_decision_tree
from funcs.evaluation import calculate_metrics, create_confusion_matrix

logger = logging.getLogger('CIFAR10')

# A class to analyze how decision tree depth affects model performance.
class DecisionTreeAnalyzer:
    def __init__(self, train_features, train_labels, test_features, test_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.results = None
        self.depths = None
    
    # Function to run depth analysis
    def run_analysis(self, depths=None):
        # Default depths if none provided
        if depths is None:
            depths = [1, 2, 5, 10, 20, 50, 100]
        
        self.depths = depths
        logger.info("Starting decision tree depth analysis...")
        
        # Initialize results dictionary
        self.results = {
            'train_accuracy': [],
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': []
        }
        
        # Test each depth
        for depth in tqdm(depths, desc="Testing different tree depths"):
            # Train tree and get predictions
            tree = train_decision_tree(self.train_features, self.train_labels, max_depth=depth)
            train_pred = predict_decision_tree(self.train_features, tree)
            test_pred = predict_decision_tree(self.test_features, tree)
            
            # Calculate metrics
            train_metrics = calculate_metrics(create_confusion_matrix(self.train_labels, train_pred))
            test_metrics = calculate_metrics(create_confusion_matrix(self.test_labels, test_pred))
            
            # Store results
            self.results['train_accuracy'].append(train_metrics['accuracy'])
            self.results['test_accuracy'].append(test_metrics['accuracy'])
            self.results['test_precision'].append(test_metrics['avg_precision'])
            self.results['test_recall'].append(test_metrics['avg_recall'])
            self.results['test_f1'].append(test_metrics['avg_f1'])
        
        logger.info("Depth analysis complete!")
    
    # Definition to create and save analysis results graphs
    def plot_results(self, save_dir=None):
        if self.results is None:
            raise ValueError("No results to plot. Run analysis first.")
        
        plt.figure(figsize=(12, 6))
        plt.suptitle('Decision Tree Depth Analysis', fontsize=16, y=0.98)
        plt.subplots_adjust(hspace=0.5, wspace=0, left=0.1, right=0.9, top=0.90, bottom=0.05)
        
        # Plot training vs testing accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.depths, self.results['train_accuracy'], 'b-', label='Training Accuracy')
        plt.plot(self.depths, self.results['test_accuracy'], 'r-', label='Testing Accuracy')
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Training vs Testing Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot test metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.depths, self.results['test_precision'], 'g-', label='Precision')
        plt.plot(self.depths, self.results['test_recall'], 'b-', label='Recall')
        plt.plot(self.depths, self.results['test_f1'], 'r-', label='F1 Score')
        plt.xlabel('Tree Depth')
        plt.ylabel('Score')
        plt.title('Test Set Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'decision_tree_depth_analysis.png'), 
                       bbox_inches='tight', dpi=300)
            logger.info(f"Saved depth analysis plot to {save_dir}")
        
        plt.show(block=False)
    
    # Function to display a table of metrics for each tree depth and save to file
    def display_metrics(self, save_dir=None):
        if self.results is None:
            raise ValueError("No results to display. Run analysis first.")
        
        # Prepare table strings
        header = "Decision Tree Depth Analysis:"
        divider = "-" * 80
        column_headers = (f"{'Depth':<8} {'Train Acc':>10} {'Test Acc':>10} "
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
        
        # Print to console
        logger.info("\n" + table_content)
        
        # Save if directory provided
        if save_dir:
            save_path = os.path.join(save_dir, 'dt_depth_metrics.txt')
            with open(save_path, 'w') as f:
                f.write(table_content)
            logger.info(f"Saved DT depth analysis metrics to {save_path}")