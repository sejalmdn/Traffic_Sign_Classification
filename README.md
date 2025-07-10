# German Traffic Sign classification using CNN
This project implements a simple Convolutional Neural Network (CNN) model to classify german traffic signs using the [**GTSRB (German Traffic Sign Recognition Benchmark)**](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data) dataset. The goal is to classify the images into one of the 43 classes using Tensorflow and Keras.

## Project Overview
All the implementation details are available in the Jupyter notebook. Below is a brief summary of the steps performed:
* **Dataset**
  * Loaded the dataset from [GTSRB, Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data).
  * Visualized Class Distribution of each of the 43 classes using bar plots
  * Preprocessed the data by normalizing and image resizing
  * Converted labels to one-hote encoded format
* **Train-Validation Split and Data Augmentation**
  * Split the data into training and validation data
  * Applied random transformations on the training data using Keras `ImageDataGenerator`
* **Model Building**
  * Built a simple CNN model with Conv2D, Maxpooling, Dropout and Dense layers
  * Used `softmax` activation in the ouput layer
* **Training**
  * Trained the model for 15 epochs using the augmented data and with validation data to monitor
* **Evaluation**
  * Evaluated the model using accuracy and loss curves
  * Achieved **97%** accuracy on the test set
* **Inference**
  * Saved the model and used it to predict on a local image
 
## Results
### Accuracy score on test set: 97% 

##### Accuracy during training:
![Accuracy Curve](https://github.com/sejalmdn/TS-Recog/blob/main/accuracy.png)
<br>
##### Loss during training:
![Loss Curve](https://github.com/sejalmdn/TS-Recog/blob/main/loss.png)
<br>
##### Confusion Matrix:
![Confusion Matrix](https://github.com/sejalmdn/TS-Recog/blob/main/confusionMatrix.png)

## Predicting on New Images
To test the model on locally stored images, use [this](https://github.com/sejalmdn/TS-Recog/blob/main/Test_Model.ipynb). <br>
Shown below is an example of the output obtained when the model is used to predict the label of a locally stored image. <br>
![test_img_pred](https://github.com/sejalmdn/TS-Recog/blob/main/test_img_pred.png)

## Set up and installation
Install the required Python packages using:
```bash
pip install -r requirements.txt
