# MNIST Handwritten Digit Classification
![image](https://user-images.githubusercontent.com/39009087/230365124-0fe4a80b-3c36-4e17-8fa6-2297d145bd95.svg)

This repository contains a Notebook (`mnist-classification.ipynb`) that demonstrates how to train a neural network model for handwritten digit classification using the MNIST dataset. The code is written in Python and utilizes TensorFlow and Keras libraries.

The MNIST dataset is a popular benchmark dataset in the field of machine learning. It consists of 60,000 training images and 10,000 testing images of handwritten digits (0 to 9). The goal is to train a neural network model that can accurately classify the digits based on the provided images.

## Code Overview

The Notebook (`mnist-classification.ipynb`) contains the following sections:

1. **Data Loading and Preprocessing**: This section loads the MNIST dataset using the `keras.datasets.mnist.load_data()` function and performs preprocessing by normalizing the pixel values to a range of 0 to 1.

2. **Model Architecture**: The notebook defines a sequential model using Keras. The model consists of a flatten layer to convert the 2D images to 1D, followed by dense layers with ReLU activation functions. The output layer uses the softmax activation function for multi-class classification.

3. **Model Compilation**: The model is compiled with the appropriate loss function, optimizer, and metrics using the `model.compile()` function. In this case, the sparse categorical cross-entropy loss, Adam optimizer, and accuracy metric are used.

4. **Model Training**: The model is trained on the training data using the `model.fit()` function. The training is done for a specified number of epochs, with a validation split of 20% to monitor the model's performance during training. The training progress, including loss and accuracy values, is recorded.

5. **Model Evaluation**: After training, the model is evaluated on the test data by making predictions using the `model.predict()` function. The predicted probabilities are converted to discrete class labels, and the accuracy of the model is calculated using the `accuracy_score()` function from scikit-learn.

6. **Training Progress Visualization**: The loss values during training are plotted using `matplotlib.pyplot` to visualize the model's training progress. Separate plots are generated for the training loss and validation loss.

## How to Use

To run the code in the Jupyter Notebook, follow these steps:

1. Install the required dependencies: TensorFlow, Keras, NumPy, Matplotlib, and scikit-learn.

2. Download the Jupyter Notebook (`mnist-classification.ipynb`) and open it in a Jupyter Notebook environment.

3. Run each cell in the notebook sequentially to execute the code step-by-step.

4. Observe the training progress and the final accuracy of the model on the test data.

You can modify the code and experiment with different model architectures, optimization algorithms, hyperparameters, and evaluation metrics to further improve the model's performance.



## Acknowledgments

This code is inspired by the TensorFlow documentation and examples. The MNIST dataset used in this project is publicly available and widely used for benchmarking machine learning algorithms.

## References

1. [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
2. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
3. [Keras Documentation](https://keras.io/api/)
4. [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
5. 
