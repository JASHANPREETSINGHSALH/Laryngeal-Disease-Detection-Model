# Laryngeal Cancer Detection

This repository contains the implementation of various approaches for the initial stage detection of laryngeal cancer using effective hybrid features and ensemble learning methods. The project aims to achieve high accuracy in detecting laryngeal cancer from medical images using different machine learning and deep learning techniques.

## Table of Contents
- [Dataset](#dataset)
- [Approaches](#approaches)
  - [1. NN-Approach using DenseNet201](#1-nn-approach-using-densenet201)
  - [2. Ensemble Learning](#2-ensemble-learning)
  - [3. Neural Network Approach](#3-neural-network-approach)
- [Additional Experiments](#additional-experiments)
- [Results](#results)

## Dataset
The dataset used for this project is the [Laryngeal Dataset](https://zenodo.org/records/1003200). This dataset contains medical images that are used to train and evaluate the models for detecting laryngeal cancer.

## Approaches

### 1. NN-Approach using DenseNet201
1. Load the dataset.
2. Apply Gaussian Filter.
3. Split the dataset into train and test sets.
4. Import the DenseNet201 model.
5. Drop the inbuilt last classification layer.
6. Add additional layers:
   - Average Max Pooling layer
   - Dense (Fully connected layer with 1024 neurons)
   - Fully connected softmax layer with 4 neurons for the final classifications.
7. Unfreeze the last 5 layers so their weights get updated during training.
8. Train the model using StratifiedKFold cross-validation.
9. Achieve an accuracy of 99.24% on unseen data.

### 2. Ensemble Learning
1. Load the dataset.
2. Apply Median filter and then use Gaussian Filter to remove noise.
3. Display some images before and after filtering.
4. Stack features extracted from four pre-trained CNN models (VGG16, InceptionV3, DenseNet201, and EfficientNetB0) – Total: 46,976 features.
5. Split into train and test sets.
6. Normalize the input for train and test sets.
7. Train the base models (Weak Learners): Random Forest Classifier, Support Vector Classifier, and K-neighbours Classifier.
8. Stack the predictions for the train set of each classifier.
9. Repeat the same for the test set.
10. Use Logistic Regression as the final meta classifier, trained on the stacked features.
11. Achieve an accuracy of 99.2% on unseen data.
12. Provide a classification summary and the confusion matrix.

### 3. Neural Network Approach
1. Load the dataset.
2. Apply Median filter and then use Gaussian Filter to remove noise.
3. Display some images before and after filtering.
4. Stack features extracted from four pre-trained CNN models (VGG16, InceptionV3, DenseNet201, and EfficientNetB0) – Total: 46,976 features.
5. Split into train and test sets.
6. Normalize the input for train and test sets.
7. Extract the top 200 features from the 46,976 features.
8. Train a neural network with 6 layers (5 dense layers and 1 SoftMax layer).
9. Apply batch normalization and dropout techniques.
10. Tune the neural network.
11. Achieve an accuracy of 99.1% on unseen data.
12. Provide a classification summary and the confusion matrix.

## Additional Experiments
1. CLAHE Technique + NN-Approach ------ Accuracy: 90%
2. Gamma Correction + NN-Approach ------ Accuracy: 97%
3. Image Sharpening technique + NN-Approach ------ Accuracy: 96%
4. NN-Approach without feature selection (46,976 features) ------ Accuracy: 98.8%
5. Feature Extraction [Local Binary Pattern + STAT features (Mean + Standard Deviation) + DenseNet Features] ------ Accuracy: 96%
6. Recursive Feature Elimination using Random Forest (computationally expensive, not implemented)

## Results
- NN-Approach using DenseNet201: 99.24%
- Ensemble Learning: 99.2%
- Neural Network Approach: 99.1%
   
