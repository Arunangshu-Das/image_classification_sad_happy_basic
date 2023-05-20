# Convolutional Neural Network (CNN) for Image Classification

This repository contains code for a Convolutional Neural Network (CNN) model implemented using the Keras library. The model is designed for image classification tasks and consists of convolutional, max pooling, flattening, and dense layers.

## Model Architecture

The CNN model architecture is as follows:

1. Input Layer: Accepts input images with a shape of (256, 256, 3).
2. Convolutional Layer: Applies 16 filters of size (3, 3) with a stride of 1 and ReLU activation function.
3. Max Pooling Layer: Performs max pooling with a pool size of 2x2.
4. Convolutional Layer: Applies 32 filters of size (3, 3) with a stride of 1 and ReLU activation function.
5. Max Pooling Layer: Performs max pooling with a pool size of 2x2.
6. Convolutional Layer: Applies 16 filters of size (3, 3) with a stride of 1 and ReLU activation function.
7. Max Pooling Layer: Performs max pooling with a pool size of 2x2.
8. Flattening Layer: Flattens the input into a 1-dimensional array.
9. Dense Layer: Applies 256 neurons with ReLU activation function.
10. Output Layer: Applies a single neuron with sigmoid activation function for binary classification (e.g., happy/sad).

## Training and Evaluation

The model is trained using the Adam optimizer and the binary cross-entropy loss function. The training is performed for 20 epochs with the specified training and validation datasets. The training history is stored in the `hist` variable.

After training, the loss curves are plotted using Matplotlib, showing the training and validation loss over the epochs.

The model is then evaluated using the test dataset by iterating over batches of images. The predictions are obtained using the trained model, and the accuracy, precision, recall, and F1-score metrics are computed using TensorFlow's `tf.keras.metrics` functions.

Finally, an example inference is demonstrated using a single image. The image is resized and normalized, and then the model predicts the class (happy or sad) based on a threshold of 0.5.

## Demo
[**Demo Project Link**](https://youtu.be/CJGK5wkxD_8)

## Usage

To use this code, follow these steps:

1. Prepare the dataset: Make sure you have prepared your image dataset with the appropriate directory structure and labels.
2. Preprocess the dataset: You may need to perform preprocessing steps such as resizing, normalizing, and splitting the dataset into training, validation, and testing sets.
3. Configure the model: Adjust the model architecture, optimizer, loss function, and other hyperparameters according to your specific requirements.
4. Train the model: Train the model on your training and validation datasets using the `model.fit()` function.
5. Evaluate the model: Evaluate the model's performance on the test dataset using the `model.predict()` function and compute the desired metrics.
6. Make predictions: Use the trained model to make predictions on new images by resizing and normalizing them, and then passing them through the model using the `model.predict()` function.

Feel free to customize the code to suit your specific needs and requirements.

## License

This project is licensed under the [MIT License](LICENSE).
