import os
import sys
import json

# Utility imports
from utils.dataset_loader import load_dataset
from utils.model_builder import build_model
from utils.trainer import train_model
from utils.evaluator import evaluate_model

def display_models():
    print("Available AI Models:")
    print("1. Linear Regression")
    print("2. Logistic Regression")
    print("3. Convolutional Neural Network (CNN)")
    print("4. Decision Tree")
    print("5. Transformer")

def select_model():
    display_models()
    choice = input("Select the model number you want to train (1-5): ")
    models = ["linear_regression", "logistic_regression", "cnn", "decision_tree", "transformer"]
    
    try:
        selected_model = models[int(choice) - 1]
        print(f"Selected Model: {selected_model}")
        return selected_model
    except (IndexError, ValueError):
        print("Invalid selection. Please choose a valid option.")
        sys.exit(1)

def select_dataset(model):
    print(f"Choose dataset for {model}:")
    print("1. Use default dataset")
    print("2. Upload your own dataset")
    dataset_choice = input("Select option (1 or 2): ")

    if dataset_choice == '1':
        print(f"Using default dataset for {model}...")
        dataset = load_dataset(model)
    elif dataset_choice == '2':
        dataset_path = input("Enter the path to your dataset file: ")
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            sys.exit(1)
        dataset = load_dataset(model, custom_path=dataset_path)
    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)
    
    return dataset

def get_hyperparameters():
    print("Enter hyperparameters for training:")
    epochs = int(input("Number of epochs (default 10): ") or 10)
    batch_size = int(input("Batch size (default 32): ") or 32)
    learning_rate = float(input("Learning rate (default 0.001): ") or 0.001)
    return epochs, batch_size, learning_rate

def main():
    print("SO-NAT: Systematic Optimization and Neural Architecture Training")
    
    # Select Model
    model = select_model()
    
    # Select Dataset
    dataset = select_dataset(model)
    
    # Get Hyperparameters
    epochs, batch_size, learning_rate = get_hyperparameters()
    
    # Build Model
    model_instance = build_model(model, learning_rate)
    
    # Train Model
    train_model(model_instance, dataset, epochs, batch_size)
    
    # Evaluate Model
    results = evaluate_model(model_instance, dataset)
    print(f"Model Evaluation Results: {results}")

if __name__ == "__main__":
    main()
