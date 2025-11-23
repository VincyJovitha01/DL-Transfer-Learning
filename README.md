# EX-04: DL- Developing a Neural Network Classification Model using Transfer Learning
## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## THEORY
Transfer Learning is a deep learning technique where a model pre-trained on a large dataset (such as ImageNet) is reused for a different but related task.
Instead of training a neural network from scratch, the pre-trained model’s feature extractor layers are kept and only the final classification layers are modified and trained.

VGG19 is a 19-layer deep convolutional neural network known for its simple architecture and strong feature extraction capabilities.
Because it is already trained on millions of images, it can effectively capture edges, textures, shapes, and patterns, making it ideal for image classification tasks with smaller datasets.

## Neural Network Model
Include the neural network model diagram.

## Requirements:
Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

OpenCV (optional for image reading)
## DESIGN STEPS
STEP 1: Import required libraries and define image transforms.

STEP 2: Load training and testing datasets using ImageFolder.

STEP 3: Visualize sample images from the dataset.

STEP 4: Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

STEP 5: Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

STEP 6: Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## RESULT
VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
