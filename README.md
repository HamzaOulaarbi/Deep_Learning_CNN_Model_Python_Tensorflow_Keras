# Deep_Learning_CNN_Model_Python_Tensorflow_Keras
## Image Recognition : Cats Vs Dogs  

In this project we are going to develop a Convolutional Neural Network (CNN) model to classify photos of dogs and cats. Here are the main steps : 

  - Develop Model Improvements
  - Use Tensordoard to analyze results and compare different models
### Training a Keras Sequential model with X and y as a numpy array (val_accuracy: 0.87)
  - Load and prepare photos of dogs and cats for modeling
  - Develop a basic convolutional neural network for photo classification 
### Training with a Functional Keras API (model: VGG16) (val_accuracy: 0.91)
  - Transfer Learning
  - Use the VG16 model without the tops layers (include_top=False) 
### VGG16 with all layers
  - Transfer Learning
### Training with a Functional Keras API (model: Mobilenet) ()
  -Transfer Learning
  -val_accuracy: 0.9750

## Data :
Link : https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, we predict a probability that the image is a dog (1 = dog, 0 = cat).

## CNN : 
Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. the principale tricks of this model are : 
  - Convolution layer : Convolving an image with a bunch of filters (features/convolution Kernels) and create a stack of filtred image (we get as many filtered images as we have of filters)
  - Pooling layers : used to decrease the spatial size of the convolved feature, in order to reduce the required computational power to process the data. It is useful to extract dominant features. There are two types of Pooling: Max Pooling and Average Pooling. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. Average Pooling returns the average of all the values from the portion of the image covered by the Kernel.
  - Normalization :  Rectified Linear Units (RelUs) : change every thing negative to zero (a stack of images becomes a stack of images with no negative values)

The convolution and pooling layers  form one layers of a CNN. Depending on the complexities in the images, we can add more layers for capturing low-level details even further, but at the cost of more computational power.

This layers enable the model to understand the features. The last step is to flatten the final output and feed it to a regular Neural Network for classification purposes.
