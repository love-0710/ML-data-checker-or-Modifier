## ML Data Checker and Modifier

This repository contains scripts and tools designed to facilitate the cross-checking of data in machine learning projects.
## Overview

In the realm of machine learning, ensuring the quality and integrity of the data used for training models is crucial for achieving reliable results. This repository provides a collection of simple yet effective code snippets and utilities aimed at assisting in the validation and verification of data sets.

## Features

- Data validation scripts to check for inconsistencies, missing values, and outliers.
- Data visualization tools for exploratory data analysis (EDA) and pattern identification.
- Utilities for data preprocessing, including normalization, scaling, and feature engineering.
- Cross-validation techniques and performance evaluation metrics for assessing model accuracy.
- Integration with popular machine learning libraries such as scikit-learn and TensorFlow.

## List of Code Files

- [MFCC-Abstraction-along_with-Metadata_&_captions-Extraction](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/MFCC-Abstraction-and-Metadata-Extraction.py) : This script contains code that abstracts Mel-Frequency Cepstral Coefficients (MFCC) from audio files and extracts metadata and captions from CSV files. The extracted MFCC data is then saved along with their corresponding captions and metadata into a JSON format. This codebase is useful for Task 6 of the DCASE 2023(Detection and Classification of Acoustic Scenes and Events) challenge, providing a solution for processing audio data and associated metadata efficiently.
  
- [Checking dataset Cointain ROI or not](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/dataset_roi_check.ipynb) : Script for checking if the dataset contains regions of interest (ROIs) or not.

- [Convert Keras Model to TensorFlow Lite](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/convert_keras_model_to_tflite.py): Convert a pre-trained Keras model to TensorFlow Lite format, which is suitable for deployment on resource-constrained devices such as mobile phones, IoT devices, and edge devices.

- [TensorFlow Lite Inference](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/tflite_inference.py) : This code snippet demonstrates how to perform inference using a TensorFlow Lite model. 

- [Resize Input Size of TensorFlow Model](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/changing_model_input_size.py) : This code snippet demonstrates how to resize the input size of a TensorFlow model. It loads a saved model, modifies the input layer to the desired input shape, and creates a new model with the updated input size. 

- [Checking the overall quality of saved model](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/checking_overall_quality_of_saved_model.py) : Script for checking the quality of the model.

- [Split Dataset into Train, Test, and Validation](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/spliting_dataset.py) : This code snippet demonstrates how to split a dataset into training, testing, and validation sets. It loads a dataset and divides it into specified proportions for training, testing, and validation. The resulting splits are then saved or used for further processing.

- [Rename Files in Dataset](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/rename.py) : This code snippet demonstrates how to rename files within a dataset by adding prefixes based on folder names. It iterates through the directories of a dataset, generates unique filenames with prefixes, and renames the files accordingly. This can be useful for organizing and labeling files within a dataset.
- [Agmentation-ImageDataGenerator](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/Agumentation_image.py)  : This code snippet demonstrates how to use TensorFlow's ImageDataGenerator to perform data augmentation and preprocessing for image datasets. It defines settings for generating augmented images for the training set and creates generators for the training, validation, and test sets. 
- [MFCC-Data-Augmentation](https://github.com/love-0710/ML-data-checker-or-Modifier/blob/main/MFCC_Data_Augumenattion.py) : This Script contains code for augmenting Mel-Frequency Cepstral Coefficients (MFCC) data stored in a JSON file. 
## Usage

To use the scripts and tools provided in this repository, follow these steps:

## Clone the repository to your local machine:

    git clone https://github.com/love-0710/ML-data-checker-or-Modifier.git

## Navigate to the repository directory:

    cd ML-data-checker-or-Modifier

- Explore the various directories and files to find the code snippets or utilities that suit your needs.

- Follow the instructions provided in the individual scripts or README files to execute the code and analyze your data.

## Contribution

Contributions to this repository are welcome! If you have additional scripts, tools, or improvements that could benefit the machine learning community, feel free to submit a pull request. Please ensure that your contributions adhere to the repository's coding standards and guidelines.

## License

This repository is licensed under the MIT License. See the LICENSE file for more information.
