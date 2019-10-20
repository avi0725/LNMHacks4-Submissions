Pneumonia Detection from Chest X-Ray Images using Transfer Learning
Domain             : Machine Learning
Sub-Domain         : Deep Learning, Image Recognition
Techniques         : Deep Convolutional Neural Network
Application        : Image Recognition, Image Classification, Medical Imaging
Pre-requisites     : Pyhton 3.0 and above,Spyder IDE,Keras

Description
1. Detected Pneumonia from Chest X-Ray images using Custom Deep Convololutional Neural Network with 5216 images of X-ray .
2. This is going to be a CNN using few conv layers and pooling layers to classify X-ray as 1)Normal 2)Pneumonia.
   The reason for keeping it simple is to explore parameter tunning and regularization as well.
   We will develop model using Keras. 
   We shall start with loading data normally, the preprocessing (if required) and then finding paramters using smaller datasets.
   Along with that regularization techniques such as Dropout would also be used.
3. With Custom Deep Convololutional Neural Network attained testing accuracy 88.29% and loss0.41.

Dataset
Dataset Name     : Chest X-Ray Images (Pneumonia)
Dataset Link     : Chest X-Ray Images (Pneumonia) Dataset (Kaggle)
          
Original Paper   : Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning
                   (Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, M. Anthony Lewis, Huimin Xia, Kang Zhang)
                   https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Dataset Details
Dataset Name            : Chest X-Ray Images (Pneumonia)
Number of Class         : 2
Number/Size of Images   : Total      : 5856 
                          Training   : 5216 
                          Validation : 16  
                          Testing    : 624 
Model Parameters
Machine Learning Library: Keras
Base Model              : Custom Deep Convolutional Neural Network
Optimizers              : Adam
Loss Function           : binary_crossentropy

For Custom Deep Convolutional Neural Network : 
Training Parameters
Batch Size              : 32
Number of Epochs        : 10
Training Time           : 20 Mins

Output (Prediction/ Recognition / Classification Metrics)
Testing
Accuracy (F-1) Score    : 88.29%
Loss                    : 0.41
