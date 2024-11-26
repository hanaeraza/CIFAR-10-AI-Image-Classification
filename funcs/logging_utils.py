import logging
import time
import os
from tqdm import tqdm
import sys

# Function to create a new timestamped directory for the current run
def setup_run_directory():
    # Create base runs directory if it doesn't exist
    if not os.path.exists('runs'):
        os.makedirs('runs')
        
    # Create timestamped subdirectory for this run
    timestamp = time.strftime("%b%d_%H-%M")
    run_dir = os.path.join('runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

# Function to set up logging configuration
def setup_logging(run_dir):
    start_time = time.time()
    
    def elapsed_time():
        return f"[{tqdm.format_interval(time.time() - start_time)}]"
    
    class TqdmFormatter(logging.Formatter):
        def format(self, record):
            return f"{elapsed_time()} {record.getMessage()}"
    
    log_file = os.path.join(run_dir, 'logs.log')
    
    # Create logger
    logger = logging.getLogger('CIFAR10')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = TqdmFormatter()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Function to display metrics of all models in log file and in terminal
def display_metrics_table(metrics_dict, save_path):
    # Prepare table strings
    header = "Evaluation Metrics:"
    divider = "-" * 80
    column_headers = (f"{'Algorithm':<20} {'Accuracy':>12} {'Precision':>12} "
                     f"{'Recall':>12} {'F1':>12}")
    
    # Build table rows
    table_rows = []
    for algo_name, metrics in metrics_dict.items():
        row = (f"{algo_name:<20} "
               f"{metrics['accuracy']:>12.3f} "
               f"{metrics['avg_precision']:>12.3f} "
               f"{metrics['avg_recall']:>12.3f} "
               f"{metrics['avg_f1']:>12.3f}")
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
    logger = logging.getLogger('CIFAR10')
    logger.info("\n" + table_content)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(table_content)
    logger.info(f"Saved model metrics to {save_path}")