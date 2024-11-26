import os
import pickle
import joblib
import logging
import shutil

logger = logging.getLogger('CIFAR10')


# Function to save a model to disk
# If custom = True, use pickle
# Else, use joblib for sklearn 
def save_model(model, model_name, custom=True, training_history=None):
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = os.path.join('models', f'{model_name}.pkl')
    history_path = os.path.join('models', f'{model_name}_history.pkl')
    
    try:
        if custom:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            joblib.dump(model, model_path)
        logger.info(f">> Saved {model_name} model to {model_path}")

        # Save training history if provided
        if training_history is not None and isinstance(training_history, dict):
            with open(history_path, 'wb') as f:
                pickle.dump(training_history, f)
            logger.info(f">> Saved {model_name} training history to {history_path}")

    except Exception as e:
        logger.error(f">> Error saving {model_name} model or training history: {str(e)}")
        raise

# Function to load model from disk
# Returns none if model file doesn't exist
def load_model(model_name, custom=True, retrain=False):

    model_path = os.path.join('models', f'{model_name}.pkl')
    history_path = os.path.join('models', f'{model_name}_history.pkl')

    # If retrain flag is True and model exists, delete it
    if retrain:
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(history_path):
            os.remove(history_path)
        logger.info(f">> Cleared {model_name} model for retraining")
        return None, None
    
    # If no model exists, return None
    if not os.path.exists(model_path):
        logger.info(f">> No saved model found for {model_name}")
        return None, None
    
    try:
        # Load model
        if custom:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = joblib.load(model_path)
        logger.info(f">> Loaded {model_name} model from {model_path}")

        # Load training history if it exists
        training_history = None
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                training_history = pickle.load(f)
            logger.info(f">> Loaded {model_name} training history from {history_path}")
        return model, training_history
    
    except Exception as e:
        logger.error(f">> Error loading {model_name} model or training history: {str(e)}")
        raise