# CIFAR-10 AI Image Classification

This project implements image classification on the CIFAR-10 dataset using Naive Bayes, Decision Tree, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN) models in Python and PyTorch. The models are evaluated using confusion matrix, accuracy, precision, recall, and f1-measure metrics. This project was developed for the course COMP 472. 

## Project Files

### Core Files

- `main.py`: Main script for data loading, preprocessing, model training, evaluation, and visualization
- `funcs/extract_features.py`: Functions for extracting features from CIFAR-10 images using ResNet18
- `funcs/naive_bayes.py`: Gaussian Naive Bayes implementation and functions for training and prediction
- `funcs/decision_tree.py`: Decision Tree implementation and functions for training and prediction
- `funcs/mlp.py`: Multi-Layer Perceptron (MLP) model definition, training, and prediction functions
- `funcs/cnn.py`: Convolutional Neural Network (CNN) model definition based on VGG11

### Analysis Files

- `funcs/dt_analysis.py`: Class for analyzing how decision tree depth affects model performance
- `funcs/cnn_analysis.py`: Class for analyzing CNN depth and kernel size variations

### Utility Files

- `funcs/evaluation.py`: Functions for creating confusion matrices and calculating metrics
- `funcs/save_load_models.py`: Utility functions for saving and loading trained models
- `funcs/logging_utils.py`: Utility functions for logging, displaying metrics, and saving graphs
- `funcs/requirements.txt`: Required Python packages and versions

### Model Files

- `custom_gnb.pkl`: Custom Gaussian Naive Bayes model
- `sklearn_gnb.pkl`: Scikit-learn Gaussian Naive Bayes model
- `custom_dt.pkl`: Custom Decision Tree model
- `sklearn_dt.pkl`: Scikit-learn Decision Tree model
- `mlp.pkl`: Multi-Layer Perceptron model
- `mlp_history.pkl`: Training history for the MLP model
- `cnn.pkl`: Convolutional Neural Network model
- `cnn_history.pkl`: Training history for the CNN model

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/hanaeraza/CIFAR-10-COMP472.git
   cd CIFAR-10-COMP472
   ```

2. (Optional) Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Data Preprocessing

The CIFAR-10 dataset is automatically downloaded when running `main.py`. The preprocessing steps include:

1. Subsets the dataset to 500 training and 100 test images per class
2. Resizes images to 224x224 and normalizing them
3. Extracts features using ResNet18
4. Applies PCA to reduce feature dimensionality from 512 to 50

## Model Training and Evaluation

Run `main.py` to train and evaluate the models:

```
python main.py
```

The script will:

1. Load and preprocess the CIFAR-10 dataset
2. Train or load the models based on the configuration
3. Evaluate each model and calculate metrics
4. Create visualizations and save results

### Loading Models

To control which models are trained or loaded, modify the `RETRAIN` dictionary in `main.py`:

```python
RETRAIN = {
    'custom_gnb': False,
    'sklearn_gnb': False,
    'custom_dt': False,
    'sklearn_dt': False,
    'mlp': False,
    'cnn': False
}
```

To force retraining of a specific model, set any value to `True`. Otherwise, leave as `False` to load a previously saved model if it's available.

## Results

After running `main.py`, the results will be saved in a timestamped folder under `runs`:

- `metrics.txt`: Evaluation metrics for each model
- `confusion_matrices.png`: Confusion matrices for all models
- `mlp_metrics.png`: MLP training loss and test accuracy plots
- `cnn_metrics.png`: CNN training loss and test accuracy plots
- `dt_depth_analysis.png`: Graphs of how tree depth affects model performance
- `dt_depth_metrics.txt`: Metrics for different tree depths
- `cnn_depth_analysis.png`: Graphs of how CNN depth affects model performance
- `cnn_depth_metrics.txt`: Metrics for different CNN depths
- `kernel_size_analysis.png`: Graphs of how CNN kernel size affects model performance
- `kernel_size_metrics.txt`: Metrics for different CNN kernel sizes
- `logs.log`: Detailed logs of the training and evaluation process

The plots will also be displayed interactively using matplotlib.
