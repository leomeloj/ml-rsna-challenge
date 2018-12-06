# Introduction

This repository contains a final home work to be delivered at the end of the semester.
The focus is a kaggle chalenge for pneumony detection in X-Ray Images.

## Directory Structure

- `CheXNet` - Project based on https://www.github.com/jrzech/reproduce-chexnet.git and fine-tunned on the RSNA Dataset.
- `rsna-pneumonia-retina` - Project based on https://github.com/pmcheng/rsna-pneumonia.
- `docker-utils` - Contains the docker used to run the rsna-pneumonia-retina and CheXnet architectures.
- `kernels` - This directory contains a set of notebooks used do the exploratory analysis over the dataset, some extracted from the Kaggle competion.

### CheXNet

The repository come with a README.md describing how to create a conda environment the model to run on pytorch. We improved from that loading the pre-trained weights and freezing the inner layers for the fine-tunning.
In the directory we added notebooks to visualize predictions and run the retraining.

### rsna-pneumonia-retina

Clonned from another competitor, the code was adpated in few parts to enable the model and result analysis alongside with the training on our development environments.
