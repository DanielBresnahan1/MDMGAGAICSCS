# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:02:53 2022

@author: Daniel Bresnahan
"""
import numpy as np                       
import torch                       
import torchvision                       
from torch import nn                       
from torch.autograd import Variable                       
from torchvision.datasets import MNIST                       
from torchvision.transforms import transforms                       
from torchvision.utils import save_image                       
import matplotlib.pyplot as plt
import os
from corndataset import CornDataset


class Autoencoder(nn.Module):
    def __init__(self, datasetDir, epochs=100, batchSize=128, learningRate=1e-3):
        super(Autoencoder, self).__init__()
        # Encoder Network
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 6000, 4000),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))
        # Decoder Network
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 784),
                                     nn.Tanh())

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        
        
        csv = os.path.join(datasetDir, "annotations_handheld.csv")

        self.corn_dataset = CornDataset(csv, self.datasetDir)
        
        self.dataLoader = torch.utils.data.DataLoader(dataset=self.corn_dataset,
                                                      batch_size=self.batchSize,
                                                      shuffle=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()