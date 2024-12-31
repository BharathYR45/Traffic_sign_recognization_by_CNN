# Traffic Sign Recognition System

This project implements a **Traffic Sign Recognition** system using **deep learning** and **TensorFlow**. The goal of this project is to classify different traffic signs from images using a convolutional neural network (CNN). The system leverages the power of machine learning models to identify and interpret traffic signs to assist in autonomous driving or smart traffic management.

## Features

- **Traffic Sign Classification**: Classifies different traffic signs from images.
- **Google Colab Integration**: Uses Google Colab for training the model.

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The dataset includes over 50,000 images of 43 different traffic sign classes. Each image is labeled, making it suitable for supervised learning tasks.

- **Classes**: 43 unique traffic signs, including speed limits, warnings, and prohibitory signs.
- **Format**: Images are provided in `.ppm` format with varying dimensions.
- **link**: "https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

### Data Preprocessing

1. Images are resized to a fixed dimension (e.g., 32x32) for input into the CNN model.
2. Normalization is applied to ensure all pixel values lie between 0 and 1.
3. The dataset is split into training, validation, and test sets for model evaluation.

## Model Architecture

The system uses a Convolutional Neural Network (CNN) designed for image classification tasks. The architecture includes:

1. **Input Layer**: Takes preprocessed traffic sign images as input.
2. **Convolutional Layers**: Extracts spatial features from the images.
3. **Pooling Layers**: Reduces the spatial dimensions while retaining essential features.
4. **Fully Connected Layers**: Combines extracted features to classify the traffic signs.
5. **Output Layer**: Predicts the traffic sign class using a softmax activation function.

### Key Model Features

- **Activation Function**: ReLU is used in hidden layers, and softmax is used for the output layer.
- **Optimization**: Adam optimizer for faster convergence.
- **Loss Function**: Categorical cross-entropy for multi-class classification.
- **Regularization**: Dropout is applied to prevent overfitting.

