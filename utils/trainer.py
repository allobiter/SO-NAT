from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np

def train_model(model, dataset, epochs=10, batch_size=32):
    """
    Trains the model using the dataset.
    Args:
        model (sklearn or TensorFlow model): The model to be trained.
        dataset (DataFrame or np.array): The dataset used for training.
        epochs (int): Number of epochs for training (default 10).
        batch_size (int): The batch size (default 32).
    """
    if model.__class__ in [LinearRegression, LogisticRegression, DecisionTreeClassifier]:
        # Preprocess dataset for sklearn models
        X = dataset.iloc[:, :-1]  # Features
        y = dataset.iloc[:, -1]   # Target variable
        
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        print(f"Training {model.__class__.__name__} model...")
        model.fit(X_train, y_train)
        
        # Predict and evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy*100:.2f}%")

    elif isinstance(model, tf.keras.Model):
        # For CNN models (using TensorFlow/Keras)
        X = dataset['images']  # Assuming 'images' column holds image data
        y = dataset['labels']  # Assuming 'labels' column holds labels

        # Normalize images
        X = X / 255.0
        
        # Train the model
        print(f"Training {model.name} CNN model...")
        model.fit(X, y, epochs=epochs, batch_size=batch_size)

        # Evaluate the model
        loss, accuracy = model.evaluate(X, y)
        print(f"Model accuracy: {accuracy*100:.2f}%")

    else:
        print("Unknown model type for training.")
