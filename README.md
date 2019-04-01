## P3 - Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
Overview
---
In this project, I used what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I trained and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

I've included an Ipython notebook that contains further instructions and starter code. Be sure to download the [Ipython notebook](https://github.com/JohnBetaCode/CarND-P3-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb), and then check and follow the instructions in the [markdown file ](https://github.com/JohnBetaCode/CarND-P3-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.md). 

To meet specifications, the project has three principal files: 
* the [Ipython notebook](https://github.com/JohnBetaCode/CarND-P3-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)  with the code
* the code exported as an [html](https://github.com/JohnBetaCode/CarND-P3-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.html)  file
* a writeup report as a [markdown file](https://github.com/JohnBetaCode/CarND-P3-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.md) 

---
Writeup
---
**Dataset Exploration**:  
* Dataset Summary: The submission includes a basic summary of the data set.
* Exploratory Visualization: The submission includes an exploratory visualization on the dataset.

**Designing and Testing a Model Architecture**:
* Preprocessing: The submission describes the preprocessing techniques used and why these techniques were chosen.
* Model Architecture: The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
* Model Training: The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyper-parameters.
* Solution Approach: The submission describes the approach to finding a solution. Accuracy on the validation set is 0.97.

**Test a Model on New Images**:
* Acquiring New Images: The submission includes eight new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.
* Performance on New Images: The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.
* Model Certainty - Softmax Probabilities: The top five Softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

The Project
---
The goals / steps of this project were the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the Softmax probabilities of the new images

---
### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for details.

---
### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which  the images are resized to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/JohnBetaCode/CarND-P3-Traffic_Sign_Classifier.git
cd CarND-P3-Traffic_Sign_Classifier
jupyter notebook Traffic_Sign_Classifier.ipynb
```
