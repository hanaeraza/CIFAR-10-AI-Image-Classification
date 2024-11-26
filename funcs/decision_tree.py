import numpy as np
from tqdm import tqdm

class Node:
    def __init__(self):
        self.feature_column = None
        self.threshold = None
        self.left = None
        self.right = None
        self.predicted_class = None

# Function to calculate gini index (probability of a feature being classified incorrectly)
# 0 = perfectly pure, 1 = perfectly impure
def calculate_gini(labels):
    total_samples = len(labels)
    if total_samples == 0:
        return 0
    
    _, class_counts = np.unique(labels, return_counts=True)
    class_probabilities = class_counts / total_samples
    gini = 1 - np.sum(class_probabilities ** 2)
    return gini

# Function to find best feature and threshold to split on based on Gini impurity
def find_best_split(features, labels):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    num_features = features.shape[1]
    
    for feature_column in range(num_features):
        possible_thresholds = np.unique(features[:, feature_column])
        
        for threshold in possible_thresholds:
            left_mask = features[:, feature_column] <= threshold
            left_labels = labels[left_mask]
            right_labels = labels[~left_mask]
            
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            gini_left = calculate_gini(left_labels)
            gini_right = calculate_gini(right_labels)
            
            num_left = len(left_labels)
            num_right = len(right_labels)
            gini = (num_left * gini_left + num_right * gini_right) / (num_left + num_right)
            
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_column
                best_threshold = threshold
    
    return best_feature, best_threshold

# Function to split data into two datasets depending on feature value compared to threshold
def split_data(features, labels, feature_column, threshold):
    left_features = []
    left_labels = []
    right_features = []
    right_labels = []
    
    for image_idx in range(len(features)):
        feature_value = features[image_idx][feature_column]
        
        if feature_value <= threshold:
            left_features.append(features[image_idx])
            left_labels.append(labels[image_idx])
        else:
            right_features.append(features[image_idx])
            right_labels.append(labels[image_idx])
    
    return (np.array(left_features), np.array(right_features), 
            np.array(left_labels), np.array(right_labels))


# Function to recursively build decision tree
def build_tree(features, labels, current_depth, max_depth):
    node = Node()
    
    if current_depth >= max_depth or len(np.unique(labels)) == 1:
        node.predicted_class = np.argmax(np.bincount(labels))
        return node
    
    feature_column, threshold = find_best_split(features, labels)
    
    if feature_column is None:
        node.predicted_class = np.argmax(np.bincount(labels))
        return node
    
    left_features, right_features, left_labels, right_labels = \
        split_data(features, labels, feature_column, threshold)
    
    node.feature_column = feature_column
    node.threshold = threshold
    
    node.left = build_tree(left_features, left_labels, current_depth + 1, max_depth)
    node.right = build_tree(right_features, right_labels, current_depth + 1, max_depth)
    
    return node

# Function to build the full tree using training data
def train_decision_tree(train_features, train_labels, max_depth=50):
    root = build_tree(train_features, train_labels, current_depth=0, max_depth=max_depth)
    return root

# Function to predict one sample by traversing the tree
def predict_one_sample(features, node):
    if node.predicted_class is not None:
        return node.predicted_class
    
    if features[node.feature_column] <= node.threshold:
        return predict_one_sample(features, node.left)
    else:
        return predict_one_sample(features, node.right)

# Function to predict classes for all samples
def predict_decision_tree(features, tree):
    predictions = []
    for sample_features in features:
        prediction = predict_one_sample(sample_features, tree)
        predictions.append(prediction)
    return np.array(predictions)