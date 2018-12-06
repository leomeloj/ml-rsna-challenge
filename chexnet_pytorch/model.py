from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E
import visualize_prediction as V
from densenet121 import densenet as dn

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results/checkpoint')


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch

def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels
    #N_LABELS = 2 # In this case, we want to train for 2 class (has or not pneumonia)

    # load labels
    print("- Loading Data")
    df = pd.read_csv("rsna_labels.csv", index_col=0)

    # define torchvision transforms
    print("- Transforming Images")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    model = models.densenet121(pretrained=False)
    
    # loading cheXnet weights
    PATH_TO_MODEL = "pretrained/checkpoint"
    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model.cpu()
    
    # freezing all parameters so they wont be updated
    for param in model.parameters():
        param.requires_grad = False
        
    # fine tunning on the last layer    
    num_ftrs = model.classifier[0].in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Softmax())

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs

def fine_tunning_FC(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    
    #==========================================
    # Initialization
    #==========================================
    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 0 - Does not have pneumonia, and 1 - positive for pneumonia
    N_LABELS = 2

    #==========================================
    # Load labels
    #==========================================
    print("- Loading Data")
    df = pd.read_csv("rsna_labels.csv", index_col=0)

    #==========================================
    # Define torchvision transforms
    #==========================================
    print("- Transforming Images")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    #==========================================
    # Create train/val dataloaders
    #==========================================
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    model = models.densenet121(pretrained=False)
    
    #==========================================
    # Loading cheXnet weights
    #==========================================
    PATH_TO_MODEL = "pretrained/checkpoint"
    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model.cpu()
    
    #==========================================
    # Freezing all layers so they wont be updated
    #==========================================
    for param in model.parameters():
        param.requires_grad = False
        
    #==========================================    
    # Fine-Tunning on the last layer    
    #==========================================
    
    # calculating the input of the new layer
    num_ftrs = model.classifier[0].in_features
    
    # switching the classifier for one with only 2 classes
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Softmax())

    #==========================================
    # Put model on GPU
    #==========================================
    model = model.cuda()

    #==========================================
    # Define criterion, optimizer for training
    #==========================================
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    #==========================================
    # Train model
    #==========================================
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    #==========================================
    # Get preds and AUCs on test fold
    #==========================================
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs, model

def fine_tunning_conv(PATH_TO_IMAGES, LR, WEIGHT_DECAY, freezing_layer_threshold):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    
    #==========================================
    # Initialization
    #==========================================
    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 0 - Does not have pneumonia, and 1 - positive for pneumonia
    N_LABELS = 2

    #==========================================
    # Load labels
    #==========================================
    print("- Loading Data")
    df = pd.read_csv("rsna_labels.csv", index_col=0)

    #==========================================
    # Define torchvision transforms
    #==========================================
    print("- Transforming Images")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    #==========================================
    # Create train/val dataloaders
    #==========================================
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    model = models.densenet121(pretrained=False)
    
    #==========================================
    # Loading cheXnet weights
    #==========================================
    PATH_TO_MODEL = "pretrained/checkpoint"
    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model.cpu()
    
    #==========================================
    # Freezing all layers so they wont be updated
    #==========================================    
    
    # the layers higher than this threshold will remain trainable
    print("Freezing layers until layer",str(freezing_layer_threshold))
    # number of children
    child_counter = 0
    for child in model.children():        
        # number of children of the children
        children_of_child_counter = 0
        for children_of_child in child.children():
            # checking if this layer should be frozen
            if (children_of_child_counter < freezing_layer_threshold):
                for param in children_of_child.parameters():
                    param.requires_grad = False
            children_of_child_counter += 1
        child_counter += 1
        
    #==========================================    
    # Fine-Tunning on the last layer    
    #==========================================    
    # calculating the input of the new layer
    num_ftrs = model.classifier[0].in_features
    
    # switching the classifier for one with only 2 classes
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Softmax())

    #==========================================
    # Put model on GPU
    #==========================================
    model = model.cuda()

    #==========================================
    # Define criterion, optimizer for training
    #==========================================
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    #==========================================
    # Train model
    #==========================================
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    #==========================================
    # Get preds and AUCs on test fold
    #==========================================
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs, model


def train_full(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    
    #==========================================
    # Initialization
    #==========================================
    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 0 - Does not have pneumonia, and 1 - positive for pneumonia
    N_LABELS = 2

    #==========================================
    # Load labels
    #==========================================
    print("- Loading Data")
    df = pd.read_csv("rsna_labels.csv", index_col=0)

    #==========================================
    # Define torchvision transforms
    #==========================================
    print("- Transforming Images")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    #==========================================
    # Create train/val dataloaders
    #==========================================
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.RSNA_Dataset(
        path_to_images=PATH_TO_IMAGES,
        mode='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    model = models.densenet121(pretrained=False)
    
    #==========================================
    # Loading cheXnet weights
    #==========================================
    PATH_TO_MODEL = "pretrained/checkpoint"
    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model.cpu()
    
    #==========================================    
    # Addapting last Layer and adding Metadata
    #==========================================    
    # calculating the input of the new layer
    num_ftrs = model.classifier[0].in_features
    
    # switching the classifier for one with only 2 classes
    
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Softmax())

    #==========================================
    # Put model on GPU
    #==========================================
    model = model.cuda()

    #==========================================
    # Define criterion, optimizer for training
    #==========================================
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    #==========================================
    # Train model
    #==========================================
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    #==========================================
    # Get preds and AUCs on test fold
    #==========================================
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs, model

def train_model_meta(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, meta, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                meta = Variable(meta.cuda()).float()
                outputs = model(inputs,meta)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch

def train_full_meta(PATH_TO_IMAGES, PATH_TO_METADATA, LR, WEIGHT_DECAY):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    
    #==========================================
    # Initialization
    #==========================================
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    
    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 0 - Does not have pneumonia, and 1 - positive for pneumonia
    N_LABELS = 2
    
    # Number of metadatas used
    N_META = 4

    #==========================================
    # Load labels
    #==========================================
    print("- Loading Data")
    df = pd.read_csv("rsna_labels.csv", index_col=0)

    #==========================================
    # Define torchvision transforms
    #==========================================
    print("- Transforming Images")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


    #==========================================
    # Create train/val dataloaders
    #==========================================
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.RSNAmeta_Dataset(
        path_to_images=PATH_TO_IMAGES,
        path_to_metadata=PATH_TO_METADATA,
        mode='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.RSNAmeta_Dataset(
        path_to_images=PATH_TO_IMAGES,
            path_to_metadata=PATH_TO_METADATA,
        mode='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    
    #==========================================
    # Loading cheXnet weights
    #==========================================
    PATH_TO_MODEL = "full_train_rsna/checkpoint"
    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    kwargs = {
            'bn_size':4, 
            'drop_rate':0,
            'num_classes':2
            }
    model = dn.densenet121(False,**kwargs)    
    model_aux = checkpoint['model']           
    state_dict = model_aux.state_dict()    
    model.load_state_dict(state_dict)
    del checkpoint
    del model_aux
    model.cpu()
    
    #==========================================    
    # Addapting last Layer and adding Metadata
    #==========================================    
    # calculating the input of the new layer
    num_ftrs = model.classifier[0].in_features + N_META # sex and age
    
    # switching the classifier for one with only 2 classes
    
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Softmax())

    #==========================================
    # Put model on GPU
    #==========================================
    model = model.cuda()

    #==========================================
    # Define criterion, optimizer for training
    #==========================================
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    #==========================================
    # Train model
    #==========================================
    model, best_epoch = train_model_meta(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    #==========================================
    # Get preds and AUCs on test fold
    #==========================================
    preds, aucs = E.make_pred_multilabel_meta(
        data_transforms, model, PATH_TO_IMAGES, PATH_TO_METADATA)

    return preds, aucs, model
