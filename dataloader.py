"""
 multi-gpu, multi-server distributed learning test using pytorch DDT
 test case: a basic classifier processing  MNIST
 author: georand
 source: https://github.com/georand/distributedpytorch
 date: 2021
"""

import os
import copy
import logging

import torch
import torch.utils.data as PTdata
import torchvision.transforms as PTVtransforms
import torchvision.datasets as PTVdatasets
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np

from config import *

VALID_RATIO = 0.9
TEST_RATIO = 0.15

##############################################################################"
def tensorRepeat(data):
  return data.unsqueeze_(0).repeat(3, 1, 1)


class TensorRepeat(torch.nn.Module):
  '''
  duolicate a 1D tensor into N channels (grayscale to rgb for instance)
  code derived from https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
  '''
  def __init__(self, num_output_channels=1):
    super().__init__()
    self.num_output_channels = num_output_channels

  def forward(self, tensor):
    return tensor.repeat(self.num_output_channels, 1, 1)

  def __repr__(self):
    return self.__class__.__name__ + f"(num_output_channels={self.num_output_channels})"


class MNIST():
  def __init__(self, dataPath, batchSize, dataDim, worldSize, rank, gpu):

    self.dataPath = dataPath
    self.batchSize = batchSize
    self.dataDim = dataDim
    self.worldSize = worldSize
    self.rank = rank
    self.gpu = gpu

    self.nbClasses = 10

    self.data = {'train':None,'valid':None,'test':None}
    self.sampler = {'train':None,'valid':None,'test':None}
    self.iterator = {'train':None,'valid':None,'test':None}

    self.mean = self.std = 0

    self.loadDataset()
    self.initIterator()

    # display log info only for the first gpu of the master node
    if self.gpu <= 0:
      logging.info(f"Number of training examples: {len(self.data['train'])}")
      logging.info(f"Number of validation examples: {len(self.data['valid'])}")
      logging.info(f"Number of testing examples: {len(self.data['test'])}")

  def __getstate__(self):

    # this method is called when you are
    # going to pickle the class, to know what to pickle
    state = self.__dict__.copy()

    return state

  def __setstate__(self, state):
    self.__dict__.update(state)

  def downloadDataset(self):
    logging.info('retrieving data')
    PTVdatasets.MNIST(root = self.dataPath, train = True, download = True)

  def loadDataset(self):

    # compute mean/std
    data = PTVdatasets.MNIST(root = self.dataPath,
                                  train = True, download = False)
    self.mean = data.data.float().mean() / 255
    self.std = data.data.float().std() / 255
    del data

    # define transformations
    # use custom TensorRepeat(3) for Grayscale->RGB instead of lambda function
    # since lambda prevents pickle serialization required by  mulitasking (if num_worker>0)
    trainTransforms = PTVtransforms.Compose([
                        PTVtransforms.RandomRotation(5, fill=(0,)),
                        PTVtransforms.RandomResizedCrop(self.dataDim),
                        PTVtransforms.ToTensor(),
                        # PTVtransforms.Lambda(lambda x: x.repeat(3, 1, 1), ),
                        TensorRepeat(3),
                        PTVtransforms.Normalize(mean = [self.mean],
                                                std = [self.std])])
    testTransforms = PTVtransforms.Compose([
                       PTVtransforms.Resize(self.dataDim),
                       PTVtransforms.CenterCrop(self.dataDim),
                       PTVtransforms.ToTensor(),
                       # PTVtransforms.Lambda(lambda x: x.repeat(3, 1, 1), ),
                       TensorRepeat(3),
                       PTVtransforms.Normalize(mean = [self.mean],
                                              std = [self.std])])
    # load data
    self.data['train'] = PTVdatasets.MNIST(root = self.dataPath,
                                           train = True,
                                           download = False,
                                           transform = trainTransforms)

    self.data['test'] = PTVdatasets.MNIST(root = self.dataPath,
                                          train = False,
                                          download = False,
                                          transform = testTransforms)

    # extract valid data from train data
    nTrainExamples = int(len(self.data['train']) * VALID_RATIO)
    nValidExamples = len(self.data['train']) - nTrainExamples
    self.data['train'], self.data['valid'] = PTdata.random_split(self.data['train'],
                                                                 [nTrainExamples,
                                                                  nValidExamples])
    self.data['valid'] = copy.deepcopy(self.data['valid'])
    self.data['valid'].dataset.transform = testTransforms

  def initIterator(self):

    if DEBUG:
      # for a quick test using only  200 samples
      subsetRange = list(range(0, 200, 1))
      self.data['train']  = PTdata.Subset(self.data['train'], subsetRange)
      #self.data['valid']  = PTdata.Subset(self.data['valid'], subsetRange)
      #self.data['test']   = PTdata.Subset(self.data['test'], subsetRange)

    if self.gpu >= 0: # CUDA distributed
      self.sampler['train'] = PTdata.distributed.DistributedSampler(self.data['train'], shuffle=True,
                            num_replicas=self.worldSize, rank=self.rank)
      self.sampler['valid']  = PTdata.distributed.DistributedSampler(self.data['valid'], shuffle=True,
                            num_replicas=self.worldSize, rank=self.rank)
      self.sampler['test']  = PTdata.distributed.DistributedSampler(self.data['test'], shuffle=True,
                            num_replicas=self.worldSize, rank=self.rank)
      self.iterator['train'] = PTdata.DataLoader(self.data['train'],
                                           batch_size=self.batchSize,
                                           shuffle=False,
                                           num_workers=NUM_WORKERS,
                                           pin_memory=True,
                                           sampler=self.sampler['train'])
      self.iterator['valid'] = PTdata.DataLoader(self.data['valid'],
                                           batch_size=self.batchSize,
                                           shuffle=False,
                                           num_workers=NUM_WORKERS,
                                           pin_memory=True,
                                           sampler=self.sampler['valid'])
      self.iterator['test']  = PTdata.DataLoader(self.data['test'],
                                           batch_size=self.batchSize,
                                           shuffle=False,
                                           num_workers=NUM_WORKERS,
                                           pin_memory=True,
                                           sampler=self.sampler['test'])
    else: # No CUDA, just cpu
      self.iterator['train'] = PTdata.DataLoader(self.data['train'],
                                             batch_size=self.batchSize,
                                             shuffle=False)
      self.iterator['valid'] = PTdata.DataLoader(self.data['valid'],
                                             batch_size=self.batchSize,
                                             shuffle=False)
      self.iterator['test'] = PTdata.DataLoader(self.data['test'],
                                             batch_size=self.batchSize,
                                             shuffle=False)
