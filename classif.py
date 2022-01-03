#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 multi-gpu, multi-server distributed learning test using pytorch DDT
 test case: a basic classifier processing  MNIST
 author: georand
 source: https://github.com/georand/distributedpytorch
 date: 2021
"""
import time, logging, os

import torch
import torch.nn as PTnn
import torch.nn.functional as PTfunctional
import torch.optim as PToptim
from torch.optim import lr_scheduler
import torch.distributed as PTdist
import torchvision

import numpy as np

from config import *
import utils
import dataloader

##############################################################################

def processData(gpu, phase, epoch, model, modelName, dataset, criterion, optimizer):

  epochLoss = 0
  epochAcc = 0
  lastLog = 0

  if phase == 'train':
    model.train()  # Set model to training mode
  else:
    model.eval()   # Set model to evaluate mode

  nbIters = len(dataset.iterator[phase])
  i = 0
  for (inputs, labels) in dataset.iterator[phase]:
    if gpu >=0:
      inputs = inputs.cuda(non_blocking=True)
      labels = labels.cuda(non_blocking=True)
    # zero the parameter gradients
    if phase == 'train':
      optimizer.zero_grad()
    with torch.set_grad_enabled(phase == 'train'):
      if modelName == 'inception':
        outputs, aux_outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4*loss2
      else:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
      acc, _ = utils.calculateAccuracy(outputs, labels)
      if phase == 'train':
        loss.backward()
        optimizer.step()
      epochLoss += loss.item()
      epochAcc += acc.item()
    if gpu <= 0 and phase == 'train':
      n = i/nbIters*100
      print(f'\r{epoch:03d} {n:.0f}%',end='\r')
      if i and n//10 > lastLog:
        lastLog = n//10
        logging.info(f'\repoch:{epoch:03d} nb batches:{i+1:04d} mean train loss:{epochLoss/i:.5f}')
    i+=1

  return epochLoss / nbIters, epochAcc / nbIters

##############################################################################

def train(gpu, args):
  '''
  if no cuda: gpu = -1
  '''
  utils.initializeLogging()

  # initialize the Distributed Data Parallelism (DDP) process group
  rank = args.firstLocalRank + gpu
  if gpu >= 0:
    logging.info(f'local rank: {gpu}, overall rank: {rank}, world size: {args.worldSize}')
    logging.info(f'batch size: {BATCH_SIZE}, number of workers: {NUM_WORKERS}')
    PTdist.init_process_group(backend='nccl', init_method='env://',
                              world_size=args.worldSize, rank=rank)
  # initialize the random generator
  utils.setRandomSeed(SEED)

  # set model name
  if args.checkpointFile:
    modelName = GetCheckpointModelName(args.checkpointFile)
  else:
    modelName = MODEL_NAME

  # load data and create DDP iterators
  dataset = dataloader.MNIST(DATA_PATH, args.batchSize,
                             utils.getModelInputSize(modelName),
                             worldSize=args.worldSize, rank=rank, gpu=gpu)

  # initialize the model
  model, _  = utils.getModel(modelName, dataset.nbClasses,
                             FEATURE_EXTRACT, USE_PRETRAINED)
  # set the loss
  criterion = PTnn.CrossEntropyLoss()

    # set the loss
  if LOSS == 'cross_entropy':
    criterion = PTnn.CrossEntropyLoss()
  elif LOSS == 'weighted_cross_entropy':
    criterion = PTnn.CrossEntropyLoss(dataset.data['train'].classWeights)
  elif LOSS == 'focal_loss':
    lossWeights = dataset.data['train'].classWeights.to(device=gpu)
    if gpu >=0:
      lossWeights = lossWeights.to(device=gpu)
    criterion = utils.FocalLossN(lossWeights)
  else:
    logging.error("Invalid loss, exiting...")
    exit()

  # set the optimizer
  if OPTIMIZER == "adam":
    optimizer = PToptim.Adam(model.parameters(), lr=0.001)
  elif OPTIMIZER == "SGD":
    optimizer = PToptim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # If SGD, decay LR by a factor of 0.1 every 1 epochs
    scheduler = PToptim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
  else:
    logging.error("Invalid optimizer, exiting...")
    exit()

  # distribute the model and put loss in cuda
  if gpu >= 0:
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # parrallelize the model
    model = PTnn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = criterion.cuda(gpu)

  # load model from a checkpoint file if it is provided
  if args.checkpointFile:
    startEpoch, bestValidLoss = utils.loadCheckpoint(gpu, args.checkpointFile,
                                                     model, optimizer)
  else:
    startEpoch = 0
    bestValidLoss = float('inf')

  startTime = time.monotonic()

  for epoch in range(startEpoch, args.nbEpochs):

    if gpu <= 0:
      print(f'====================== epoch{epoch+1:4d} ======================')
    startEpochTime = time.monotonic()

    trainLoss, trainAcc = processData(gpu, 'train', epoch, model, modelName,
                                      dataset,criterion, optimizer)
    validLoss, validAcc = processData(gpu, 'valid', epoch, model, modelName,
                                      dataset,criterion, optimizer)
    # Required in distributed mode otherwise the batchs will be identical
    # accross the epochs
    # need also shuffle=true is required  when creating the sampler
    if gpu >=0:
      dataset.sampler['train'].set_epoch(epoch)

    # update learning rate (only for SGD)
    if OPTIMIZER == "SGD":
      scheduler.step()

    endTime = time.monotonic()
    epochMins, epochSecs = utils.getDuration(startEpochTime, endTime)
    mins, secs = utils.getDuration(startTime, endTime)

    # logging and checkpoint save
    if gpu <= 0:

      logging.info(f"{'*' if validLoss < bestValidLoss else ' '} Epoch: {epoch+1:03}  | Duration: {epochMins:03d}m {epochSecs:02d}s  | Overall duration: {mins/60:.2f}h")
      logging.info(f"  Train       | Loss: {trainLoss:.5f}       | Acc: {trainAcc*100:.2f}%")
      logging.info(f"  Validation  | Loss: {validLoss:.5f}       | Acc: {validAcc*100:.2f}%")

      previousCPFile = f'{RSL_PATH}/checkpoint-mnist-{epoch-1:03d}.pt.tar'
      if os.path.exists(previousCPFile):
        os.remove(previousCPFile)
      utils.saveCheckpoint(
        f'{RSL_PATH}/checkpoint-mnist-{modelName}-{epoch:03d}.pt.tar',
        modelName, model, optimizer, epoch, bestValidLoss)
      if validLoss < bestValidLoss:
        bestValidLoss = validLoss
        utils.saveCheckpoint(
          f'{RSL_PATH}/bestmodel-mnist-{modelName}.pt.tar',
          modelName, model, optimizer, epoch, bestValidLoss)


##############################################################################

def test(gpu, args):
  '''
  if no cuda: gpu = -1
  '''
  utils.initializeLogging()

  # initialize the Distributed Data Parallelism (DDP) process group
  rank = args.firstLocalRank + gpu
  if gpu >= 0:
    logging.info(f'local rank: {gpu}, overall rank: {rank}, world size: {args.worldSize}')
    PTdist.init_process_group(backend='nccl', init_method='env://',
                              world_size=args.worldSize, rank=rank)

  # initialize the random generator
  utils.setRandomSeed(SEED)

  # set model name
  modelName = utils.getCheckpointModelName(args.checkpointFile)

  # load data and create DDP iterators
  dataset = dataloader.MNIST(DATA_PATH, args.batchSize,
                             utils.getModelInputSize(modelName),
                             worldSize=args.worldSize, rank=rank, gpu=gpu)

  # initialize the model and make it DDP
  model,inputSize  = utils.getModel(modelName, dataset.nbClasses, FEATURE_EXTRACT, USE_PRETRAINED)
  criterion = PTnn.CrossEntropyLoss()
  if gpu >= 0:
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # parrallelize the model
    model = PTnn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion =   criterion.cuda(gpu)

  # load the model from the  checkpoint file
  utils.loadCheckpoint(gpu, args.checkpointFile, model, None)

  startTime = time.monotonic()

  # perform inference on the test subset
  loss, acc = processData(gpu, 'test', 0, model, modelName, dataset, criterion, None)

  endTime = time.monotonic()
  mins, secs = utils.getDuration(startTime, endTime)

  if gpu <= 0:
    logging.info(f'Time: {mins}m {secs}s, Acc: {acc*100:.2f}%')
