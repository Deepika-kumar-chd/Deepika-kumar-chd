# MRI-Based Brain Tumor Classification Using Deep Learning

## Introduction
Brain tumors are a significant health concern, requiring accurate and timely diagnosis for effective treatment. Traditional diagnostic methods rely on expert radiological interpretation, which can be subjective and time-consuming. Deep learning, particularly CNNs, has demonstrated success in automated medical image classification, reducing diagnostic time and improving accuracy. This project explores the application of deep learning for MRI-based brain tumor classification. Convolutional Neural Network (CNN) model was implemented and trained on an MRI dataset to classify brain scans into two categories. 

## Dataset and Processing  
Dataset was downloaded from Kaggle https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection. Dataset consist of total 253 images (155 images of scans with tumor and 98 images without tumor). 


<img src="https://deepika-kumar-chd.github.io/Deepika-kumar-chd/images/MRI-CNN/dataset.PNG" >


Preprocessing steps were essential to standardize input data and improve model performance. The preprocessing pipeline included the following:
- **Resizing**: All images were resized to a fixed dimension (256×256) to ensure uniform input size for the CNN model.
- **Rescaling**: Pixel values were normalized by rescaling them to a range of [0,1] by dividing by 255, enhancing training stability.
- **Data Augmentation**: To improve generalization and reduce overfitting, random transformations were applied, including:
  - Horizontal and vertical flipping
  - Random rotation (0.2 radians)
These preprocessing steps ensured a diverse training dataset and improved the model’s robustness against variations in MRI scans.

## Model Architecture
The CNN model was designed as a sequential deep learning network optimized for MRI classification. The architecture consists of six convolutional layers, each followed by max-pooling layers. The final dense layers use ReLU activation for feature extraction and a softmax classifier for multiclass prediction.
- **Input Layer**: Accepts images of shape (256, 256,3). It serves as the entry point for the MRI scans, preparing them for feature extraction.
- **Convolutional Layers**:
  - Conv2D: Extracts local features such as edges and textures from the input images. Deeper layers detect higher-level features (edges → shapes → complex structures). More filters (64 instead of 32) allow learning more patterns.
  - MaxPooling2D (2x2 pool size): Reduces spatial dimensions, preserving key features while decreasing computational load.
- **Flatten Layer**: Converts the extracted feature maps into a one-dimensional vector to serve as input for the dense layers.
- **Fully Connected Layers**:
  - Dense (64, ReLU activation): Captures high-level patterns and enhances feature representation.
  - Dense (n_classes, Softmax activation): Outputs probability scores for each class, facilitating final classification.


### Training and Evaluation
The model was trained for 30 epochs using the Adam optimizer with a batch size of 23. Training and validation datasets were partitioned, and performance metrics included accuracy and loss.

<img src="https://deepika-kumar-chd.github.io/Deepika-kumar-chd/images/MRI-CNN/accuracy chart.PNG" ><img src="https://deepika-kumar-chd.github.io/Deepika-kumar-chd/images/MRI-CNN/loss chart.PNG" >


### Results
The model showed steady improvement in classification performance over training epochs. Key results are summarized below:
- **Training Accuracy**: Increased from 43.9% to 78.1%
- **Validation Accuracy**: Improved from 69.5% to 82.6%
  
Final model evaluation on the test dataset yielded:
- **Test Accuracy**: 0.5294
- **Test Loss**: 75.36%
  
The implemented CNN demonstrated effective classification capability with a test accuracy of 75.36%. The inclusion of data augmentation improved generalization, while deeper layers enhanced feature extraction.


<img src="https://deepika-kumar-chd.github.io/Deepika-kumar-chd/images/MRI-CNN/result.PNG" >


## Code Implementation
For a detailed walkthrough of the code, refer to the project's [GitHub repository](https://github.com/Deepika-kumar-chd/MRI_tumor_detection_CNN).

