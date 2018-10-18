# Introduction 

This repository contains a final home work to be delivered at the end of the semester. 
The focus is a kaggle chalenge for pneumony detection in X-Ray Images. 

## Planning

 1. Convert the weights for the pretrained model to Keras / Tensorflow Backend.
 2. Enable tensorboard call backs for a better visualization of the convergence.
 3. Add class-weights and a weighted cross_entropy loss function if needed.
 4. Normalize the data only by the fraction 1/255 instead of using mean and deviation, which should be more stable during the tests.
 5. Fine-tunning fully connected layers and last convolutions.
 6. Consider the meta-data contained in the dicom files, such as age, sex and radiograph position.
 7. Data augmentation with same domain images of pneumonia (external datasets). 


