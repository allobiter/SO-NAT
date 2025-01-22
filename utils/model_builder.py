from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

def build_model(model_name, learning_rate=0.001):
    """
    Builds and returns the model based on the selected model_name.
    Args:
        model_name (str): The model selected by the user.
        learning_rate (float): The learning rate for model (default 0.001).
    Returns:
        model (sklearn or TensorFlow model): The constructed model.
    """
    if model_name == "linear_regression":
        print("Building Linear Regression model...")
        model = LinearRegression()
    
    elif model_name == "logistic_regression":
        print("Building Logistic Regression model...")
        model = LogisticRegression(max_iter=1000)
    
    elif model_name == "decision_tree":
        print("Building Decision Tree model...")
        model = DecisionTreeClassifier()
    
    elif model_name == "cnn":
        print("Building CNN model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    elif model_name == "transformer":
        print("Building Transformer model...")
        # Placeholder for building Transformer models
        model = None
    else:
        print("Unknown model.")
        model = None
    
    return model
