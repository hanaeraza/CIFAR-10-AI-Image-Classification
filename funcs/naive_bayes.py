import numpy as np

# Function for Gaussian Naive Bayes algorithm
def train_gnb(features, labels):

    # Get number of samples and features
    num_samples, num_features = features.shape
    
    # Find all unique classes
    classes = np.unique(labels)
    num_classes = len(classes)
    
    # Initialize parameters
    mean = np.zeros((num_classes, num_features))
    var = np.zeros((num_classes, num_features))
    priors = np.zeros(num_classes)
    
    # Calculate statistics for each class
    for i, c in enumerate(classes):
        # Get samples of current class
        features_class = features[labels == c]
        
        # Calculate mean and variance
        mean[i] = np.mean(features_class, axis=0)
        var[i] = np.var(features_class, axis=0) + 1e-9  # add small number to avoid division by zero
        
        # Calculate prior probability
        priors[i] = len(features_class) / num_samples
    
    # Store all parameters in a dictionary
    model_params = {
        'classes': classes,
        'mean': mean,
        'var': var,
        'priors': priors
    }
    
    return model_params

# Function to predict using trained GNB model
def predict_gnb(test_features, model_params):
    predictions = []
    
    # Get parameters from model
    classes = model_params['classes']
    mean = model_params['mean']
    var = model_params['var']
    priors = model_params['priors']
    
    # For each sample in test data
    for sample in test_features:
        class_scores = []
        
        # Calculate score for each class
        for i in range(len(classes)):
            # Calculate gaussian probability
            log_probs = -0.5 * np.log(2 * np.pi * var[i])  # normalization term
            log_probs += -0.5 * ((sample - mean[i]) ** 2) / var[i]  # exponential term
            
            # Sum up log probabilities and add log prior
            class_score = np.sum(log_probs) + np.log(priors[i])
            class_scores.append(class_score)
        
        # Pick class with highest score
        predictions.append(classes[np.argmax(class_scores)])
    
    return np.array(predictions)
