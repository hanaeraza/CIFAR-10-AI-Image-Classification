import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os

# Import my functions
from funcs.extract_features import extract_features
from funcs.naive_bayes import train_gnb, predict_gnb
from funcs.evaluation import create_confusion_matrix, calculate_metrics, plot_confusion_matrix, plot_training_metrics
from funcs.decision_tree import train_decision_tree, predict_decision_tree
from funcs.save_load_models import save_model, load_model
from funcs.mlp import train_mlp, predict_mlp
from funcs.cnn import train_cnn, predict_cnn
from funcs.logging_utils import setup_logging, setup_run_directory, display_metrics_table
from funcs.dt_analysis import DecisionTreeAnalyzer
from funcs.mlp_analysis import MLPAnalyzer
from funcs.cnn_analysis import CNNAnalyzer


# Function to make subsets of datasets
def subset_dataset(dataset, num_per_class):
    class_counts = {}
    selected_indices = []
    
    for idx, (_, label) in enumerate(dataset):
        # Only process if we still need images for this class
        if label not in class_counts:
            class_counts[label] = 0
            
        if class_counts[label] < num_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
            
        # Stop if we have enough images for all classes (0-9 for CIFAR10)
        if len(class_counts) == 10 and all(count == num_per_class for count in class_counts.values()):
            break
    
    # Sort indices to maintain deterministic order
    selected_indices.sort()
    
    return torch.utils.data.Subset(dataset, selected_indices)

def main():

    # Set to True to force retraining of model
    # Set to False to load saved trained model (Default)
    RETRAIN = {
    'custom_gnb': False,
    'sklearn_gnb': False,
    'custom_dt': False,
    'sklearn_dt': False,
    'mlp': False,
    'cnn': False
    }
    
    BATCH_SIZE = 64

    # Set up logging and saving run information
    run_dir = setup_run_directory()
    setup_logging(run_dir)
    logger = logging.getLogger('CIFAR10')
    
    # Set computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Using device: {device}')

    # Mean and std for normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Transform to resize to 224x224 and normalize
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean, std)])


    logger.info("Loading CIFAR10 dataset...")
    # Load cifar-10 training dataset and subset to 500 images per class
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = subset_dataset(trainset, 500)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Load cifar-10 test dataset and subset to 100 images per class
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testset = subset_dataset(testset, 100)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Extract features using ResNet18
    logger.info("Extracting training features...")
    train_features, train_labels = extract_features(trainloader, device)
    logger.info("Extracting testing features...")
    test_features, test_labels = extract_features(testloader, device)


    # Use PCA to reduce size from 512x1 to 50x1
    pca = PCA(n_components=50)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)

    # Training & Prediction

    # Naive Bayes Models
    logger.info("\n-- Training Gaussian Naive Bayes models --")

    # Custom GNB
    loaded = load_model('custom_gnb', custom=True, retrain=RETRAIN['custom_gnb'])
    gnb = loaded[0] if loaded is not None else None  # Only take the model, ignore history
    if gnb is None:
        logger.info("Training custom Gaussian Naive Bayes...")
        start_time = time.time()
        gnb = train_gnb(train_features, train_labels)
        train_time = time.time() - start_time
        logger.info(f">> Custom GNB training time: {train_time:.2f} seconds")

        save_model(gnb, 'custom_gnb', custom=True)
    gnb_pred = predict_gnb(test_features, gnb)
    
    # Scikit GNB
    loaded = load_model('sklearn_gnb', custom=False, retrain=RETRAIN['sklearn_gnb'])
    sk_gnb = loaded[0] if loaded is not None else None  # Only take the model, ignore history
    if sk_gnb is None:
        logger.info("Training Scikit-learn Gaussian Naive Bayes...")
        sk_gnb = GaussianNB()
        sk_gnb.fit(train_features, train_labels)

        save_model(sk_gnb, 'sklearn_gnb', custom=False)
    sk_gnb_pred = sk_gnb.predict(test_features)
    
    # Decision Tree Models
    logger.info("\n-- Training Decision Tree models --")
    
    # Custom Decision Tree
    loaded = load_model('custom_dt', custom=True, retrain=RETRAIN['custom_dt'])
    dt = loaded[0] if loaded is not None else None  # Only take the model, ignore history
    if dt is None:
        logger.info("Training custom Decision Tree...")
        start_time = time.time()
        dt = train_decision_tree(train_features, train_labels, max_depth=50)
        train_time = time.time() - start_time
        logger.info(f">> Custom Decision Tree training time: {train_time:.2f} seconds")

        save_model(dt, 'custom_dt', custom=True)
    dt_pred = predict_decision_tree(test_features, dt)
    
    # Scikit Decision Tree
    loaded = load_model('sklearn_dt', custom=False, retrain=RETRAIN['sklearn_dt'])
    sk_dt = loaded[0] if loaded is not None else None  # Only take the model, ignore history
    if sk_dt is None:
        logger.info("Training Scikit-learn Decision Tree...")
        sk_dt = DecisionTreeClassifier(max_depth=50, criterion='gini')
        sk_dt.fit(train_features, train_labels)

        save_model(sk_dt, 'sklearn_dt', custom=False)
    sk_dt_pred = sk_dt.predict(test_features)

    
    # Multi-Layer Perceptron Model
    logger.info("\n-- Training Multi-Layer Perceptron model --")

    mlp, mlp_history = load_model('mlp', custom=True, retrain=RETRAIN['mlp'])
    if mlp is None:
        logger.info("Training Multi-Layer Perceptron...")
        start_time = time.time()
        mlp, mlp_losses, mlp_accuracies = train_mlp(train_features, train_labels, test_features, test_labels, device)
        train_time = time.time() - start_time
        logger.info(f">> MLP training time: {train_time:.2f} seconds")

        # Save model with training history
        training_history = {
            'losses': mlp_losses,
            'accuracies': mlp_accuracies
        }

        save_model(mlp, 'mlp', custom=True, training_history=training_history)
    else:
        # Get training history from loaded data
        mlp_losses = mlp_history['losses'] if mlp_history else []
        mlp_accuracies = mlp_history['accuracies'] if mlp_history else []
    # Predictions    
    mlp_pred = predict_mlp(mlp, test_features, device)



    # CNN VGG11 Model
    logger.info("\n-- Training Convolutional Neural Network (VGG11) model --")

    cnn, cnn_history = load_model('cnn', custom=True, retrain=RETRAIN['cnn'])
    if cnn is None:
        logger.info("Training CNN...")
        start_time = time.time()
        cnn, cnn_losses, cnn_accuracies = train_cnn(trainloader, testloader, device) # Using train and test set directly
        train_time = time.time() - start_time
        logger.info(f">> CNN training time: {train_time:.2f} seconds")

        # Save model with training history
        training_history = {
            'losses': cnn_losses,
            'accuracies': cnn_accuracies
        }

        save_model(cnn, 'cnn', custom=True, training_history=training_history)
    else:
        # Get training history from loaded data
        cnn_losses = cnn_history['losses'] if cnn_history else []
        cnn_accuracies = cnn_history['accuracies'] if cnn_history else []

    # Predictions
    cnn_pred = predict_cnn(cnn, testloader, device)


    # Evaluation 

    # Evaluate my GNB
    logger.info("")
    logger.info("Evaluating GNB models...")
    gnb_conf_matrix = create_confusion_matrix(test_labels, gnb_pred)
    gnb_metrics = calculate_metrics(gnb_conf_matrix)
    
    # Evaluate scikit-learn GNB
    sk_gnb_conf_matrix = create_confusion_matrix(test_labels, sk_gnb_pred)
    sk_gnb_metrics = calculate_metrics(sk_gnb_conf_matrix)

    # Evaluate custom Decision Tree
    logger.info("Evaluating Decision Tree models...")
    dt_conf_matrix = create_confusion_matrix(test_labels, dt_pred)
    dt_metrics = calculate_metrics(dt_conf_matrix)
    
    # Evaluate scikit-learn DecisionTreeClassifier
    sk_dt_conf_matrix = create_confusion_matrix(test_labels, sk_dt_pred)
    sk_dt_metrics = calculate_metrics(sk_dt_conf_matrix)

    # Evaluate MLP
    logger.info("Evaluating MLP model...")
    mlp_conf_matrix = create_confusion_matrix(test_labels, mlp_pred)
    mlp_metrics = calculate_metrics(mlp_conf_matrix)

    # Evaluate CNN
    logger.info("Evaluating CNN model...")
    cnn_conf_matrix = create_confusion_matrix(test_labels, cnn_pred)
    cnn_metrics = calculate_metrics(cnn_conf_matrix)

    # Display results
    metrics_dict = {
        'Gaussian Naive Bayes': gnb_metrics,
        'Scikit GNB': sk_gnb_metrics,
        'Decision Tree': dt_metrics,
        'Scikit Decision Tree': sk_dt_metrics,
        'MLP': mlp_metrics,
        'CNN': cnn_metrics
    }
    
    # Display metrics for all the models in a table
    display_metrics_table(metrics_dict, save_path=os.path.join(run_dir, 'metrics.txt'))

    # Un-comment this block to run analysis on DT, MLP, and CNN variations ---------

    # # Analyze how DT depth affects performance
    # logger.info("\n-- Analyzing Decision Tree depth variations --")
    # dt_analyzer = DecisionTreeAnalyzer(train_features, train_labels, test_features, test_labels)
    # dt_analyzer.run_analysis()  # Uses default depths [1, 2, 5, 10, 20, 50, 100]
    # dt_analyzer.display_metrics(save_dir=run_dir)
    # dt_analyzer.plot_results(save_dir=run_dir)

    # # Analyze how MLP depth affects performance
    # logger.info("\n-- Analyzing MLP depth variations --")
    # mlp_analyzer = MLPAnalyzer(train_features, train_labels, test_features, test_labels, device)
    # mlp_analyzer.run_analysis()  # Uses default depths [1, 2, 3, 4, 5, 6]
    # mlp_analyzer.display_metrics(save_dir=run_dir)
    # mlp_analyzer.plot_results(save_dir=run_dir)

    # # Analyze how CNN depth affects performance
    # cnn_analyzer = CNNAnalyzer(trainloader, testloader, device)
    # logger.info("Analyzing CNN depth variations...")
    # cnn_analyzer.analyze_depth(num_epochs=5) # Default depths [2, 4, 6, 8, 10]
    # logger.info("Analyzing CNN kernel size variations...")
    # cnn_analyzer.analyze_kernel_sizes(num_epochs=5) # Default kernel sizes [2, 3, 5, 7]
    # cnn_analyzer.display_metrics(save_dir=run_dir)
    # cnn_analyzer.plot_results(save_dir=run_dir)

    # ------------------------------------------------------------------------------

    # Plot confusion matrices
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))

    # Figure adjustments
    fig.suptitle('Confusion Matrices for AI Models', fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.5, wspace=0, left=0.1, right=0.9, top=0.90, bottom=0.05)


    plot_confusion_matrix(ax1, gnb_conf_matrix, classes)
    ax1.set_title('Gaussian Naive Bayes')
    
    plot_confusion_matrix(ax2, sk_gnb_conf_matrix, classes)
    ax2.set_title('Scikit-learn GNB')

    plot_confusion_matrix(ax3, dt_conf_matrix, classes)
    ax3.set_title('Decision Tree')

    plot_confusion_matrix(ax4, sk_dt_conf_matrix, classes)
    ax4.set_title('Scikit-learn Decision Tree')

    plot_confusion_matrix(ax5, mlp_conf_matrix, classes)
    ax5.set_title('Multi-Layer Perceptron')

    plot_confusion_matrix(ax6, cnn_conf_matrix, classes)
    ax6.set_title('Convolutional Neural Network')

    # Save confusion matrices
    plt.savefig(os.path.join(run_dir, 'confusion_matrices.png'), bbox_inches='tight', dpi=300)
    
    # Window 2: MLP Training Metrics
    logger.info("Loading MLP training metrics...")
    plot_training_metrics(mlp_losses, mlp_accuracies, "MLP", epoch_interval=10)
    plt.savefig(os.path.join(run_dir, 'mlp_metrics.png'), bbox_inches='tight', dpi=300)
    
    # Window 3: CNN Training Metrics
    logger.info("Loading CNN training metrics...")
    plot_training_metrics(cnn_losses, cnn_accuracies, "CNN", epoch_interval=1)
    plt.savefig(os.path.join(run_dir, 'cnn_metrics.png'), bbox_inches='tight', dpi=300)
    
    plt.show()

    logger.info('Done')



if __name__ == "__main__":
    main()