
import pandas as pd
from sklearn.datasets import load_iris, load_boston
import os

def load_dataset(model_name, custom_path=None):
    if model_name == "linear_regression":
        if custom_path:
            return pd.read_csv(custom_path)
        else:
            # Load a default dataset like Boston Housing for linear regression
            data = load_boston()
            return pd.DataFrame(data=data.data, columns=data.feature_names)

    elif model_name == "logistic_regression":
        if custom_path:
            return pd.read_csv(custom_path)
        else:
            # Load a default dataset like Iris for logistic regression
            data = load_iris()
            return pd.DataFrame(data=data.data, columns=data.feature_names)

    elif model_name == "cnn":
        # For CNN, you can either load MNIST or let the user upload an image dataset
        if custom_path:
            return pd.read_csv(custom_path)
        else:
            print("For CNN, use your own image dataset or default MNIST.")
            return None

    elif model_name == "decision_tree":
        if custom_path:
            return pd.read_csv(custom_path)
        else:
            # Load a default dataset for Decision Tree
            data = load_iris()
            return pd.DataFrame(data=data.data, columns=data.feature_names)
    
    elif model_name == "transformer":
        print("Transformers require a text dataset. Please provide one.")
        return None
    else:
        print("Unknown model name.")
        return None
