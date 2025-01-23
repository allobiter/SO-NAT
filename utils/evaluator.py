from sklearn.metrics import accuracy_score

def evaluate_model(model, dataset):
    """
    Evaluates the model using the test dataset.
    Args:
        model (sklearn or TensorFlow model): The model to be evaluated.
        dataset (DataFrame or np.array): The dataset used for testing.
    Returns:
        results (dict): A dictionary of evaluation metrics.
    """
    if model.__class__ in [LinearRegression, LogisticRegression, DecisionTreeClassifier]:
        # For sklearn models
        X = dataset.iloc[:, :-1]  # Features
        y = dataset.iloc[:, -1]   # Target variable
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        results = {"accuracy": accuracy}
    
    elif isinstance(model, tf.keras.Model):
        # For CNN models (using TensorFlow/Keras)
        X = dataset['images']  # Assuming 'images' column holds image data
        y = dataset['labels']  # Assuming 'labels' column holds labels

        # Normalize images
        X = X / 255.0
        
        # Evaluate the CNN model
        loss, accuracy = model.evaluate(X, y)
        results = {"accuracy": accuracy, "loss": loss}

    else:
        results = {"error": "Unknown model type for evaluation"}
    
    return results
