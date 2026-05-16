# MNIST Handwritten Digit Classification using TensorFlow/Keras
# Project Overview

This project demonstrates how to build a Handwritten Digit Recognition System using TensorFlow and Keras with the famous MNIST dataset.

The model is trained to recognize digits from 0 to 9 using a simple Artificial Neural Network (ANN).

# Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
Dataset Information

The project uses the MNIST dataset, which contains:

Dataset	Samples
Training Images	60,000
Testing Images	10,000

Each image:

Size = 28 × 28 pixels
Grayscale image
Represents digits from 0–9
Project Workflow

# 1. Import Libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

Libraries are imported for:

Deep learning
Data visualization
Numerical operations

# 2. Load Dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

This loads the MNIST handwritten digit dataset.

# 3. Display Sample Image
plt.matshow(X_train[0])

Displays the first training image.

# 4. Data Preprocessing
Normalize Data
X_train = X_train / 255
X_test = X_test / 255

Pixel values are scaled between 0 and 1.

Flatten Images
X_train_flattened = X_train.reshape(len(X_train), 28 * 28)
X_test_flattened = X_test.reshape(len(X_test), 28 * 28)

Converts 28×28 images into 784 input features.

Model Architecture
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
# Layers Explanation
Layer	Description
Dense(100)	Hidden layer with 100 neurons
ReLU	Activation function
Dense(10)	Output layer for digits 0–9
Sigmoid	Produces probability outputs
Model Summary
Component	Value
Total Parameters	79,510
Trainable Parameters	79,510
Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Compilation Components
Component	Purpose
Adam	Optimizer
Sparse Categorical Crossentropy	Loss function
Accuracy	Evaluation metric
Training the Model
model.fit(
    X_train_flattened,
    y_train,
    epochs=5
)

The model is trained for 5 epochs.

# Training Results
Epoch	Accuracy
1	92.44%
2	96.45%
3	97.46%
4	98.04%
5	98.49%

The model accuracy improves after each epoch.

# Model Evaluation
loss, accuracy = model.evaluate(X_test_flattened, y_test)
Final Results
Metric	Value
Loss	0.0816
Accuracy	97.54%

The model performs very well on unseen test data.

# Prediction
y_predicted = model.predict(X_test_flattened)

The model predicts probabilities for each digit.

Predicted Label
predicted_label = np.argmax(y_predicted[0])

argmax() selects the digit with the highest probability.

Example:

Predicted Label : 7
Output Example
[np.int64(7), np.int64(2), np.int64(1), np.int64(0), np.int64(4)]

These are the predicted digits for the first five test images.

# Advantages of This Project
Simple and beginner-friendly
High accuracy
Fast training
Good introduction to deep learning
Applications
Handwritten digit recognition
Bank cheque processing
Postal code recognition
OCR systems
Digit classification systems

# Conclusion

This project successfully builds a handwritten digit classifier using TensorFlow and Keras.
The model achieves approximately 97.5% accuracy on the MNIST test dataset, demonstrating the effectiveness of neural networks in image classification tasks.
