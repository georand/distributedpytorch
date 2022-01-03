"""
 multi-gpu, multi-server distributed learning test using pytorch DDT
 test case: a basic classifier processing  MNIST
 author: georand
 source: https://github.com/georand/distributedpytorch
 date: 2021
"""

import os
import sys
import logging
import random

import torch
import torch.nn as PTnn
import torchvision
import torchvision.models as PTmodels
import torch.nn.functional as PTFunctional

import numpy as np

from config import *

def getModelInputSize(modelName):
  if modelName == "resnet":
    return 224
  elif modelName == "alexnet":
    return 224
  elif modelName == "vgg":
    return 224
  elif modelName == "squeezenet":
    return 224
  elif modelName == "densenet":
    return 224
  elif modelName == "inception":
    return 299

def getModel(model_name, num_classes, feature_extract, use_pretrained=False):
  model_ft = None
  input_size = 0

  if model_name == "resnet":
    """ Resnet18
    """
    model_ft = PTmodels.resnet18(pretrained=use_pretrained)
    setParameterRequiresGrad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = PTnn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == "alexnet":
    """ Alexnet
    """
    model_ft = PTmodels.alexnet(pretrained=use_pretrained)
    setParameterRequiresGrad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = PTnn.Linear(num_ftrs,num_classes)
    input_size = 224

  elif model_name == "vgg":
    """ VGG11_bn
    """
    model_ft = PTmodels.vgg11_bn(pretrained=use_pretrained)
    setParameterRequiresGrad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = PTnn.Linear(num_ftrs,num_classes)
    input_size = 224

  elif model_name == "squeezenet":
    """ Squeezenet
    """
    model_ft = PTmodels.squeezenet1_0(pretrained=use_pretrained)
    setParameterRequiresGrad(model_ft, feature_extract)
    model_ft.classifier[1] = PTnn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes
    input_size = 224

  elif model_name == "densenet":
    """ Densenet
    """
    model_ft = PTmodels.densenet121(pretrained=use_pretrained)
    setParameterRequiresGrad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = PTnn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == "inception":
    """ Inception v3
    Be careful, expects (299,299) sized images and has auxiliary output
    """
    model_ft = PTmodels.inception_v3(pretrained=use_pretrained)
    setParameterRequiresGrad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = PTnn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = PTnn.Linear(num_ftrs,num_classes)
    input_size = 299

  else:
    logging.error("Invalid model name, exiting...")
    exit()

  return model_ft, input_size

def setParameterRequiresGrad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

def saveCheckpoint(path, modelName, model, optimizer, epoch, bestValidLoss):

  torch.save({
    'model_name': modelName,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': bestValidLoss,
  }, path)
  logging.info(f'epoch:{epoch:04d}: model saved to {path}')

def loadCheckpoint(gpu, path, model, optimizer):

  if gpu>=0: # active cuda
    checkpoint = torch.load(path, map_location=f'cuda:{gpu}')
  else:
    checkpoint = torch.load(path)

  model.load_state_dict(checkpoint['model_state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']+1
  bestValidLoss = checkpoint['loss']
  logging.info(f'epoch:{epoch:04d}: model loaded from {path}')
  return epoch, bestValidLoss

def getCheckpointModelName(path):
  checkpoint = torch.load(path)
  return checkpoint['model_name']

class FocalLossN(PTnn.Module):
  # focal loss for N classes
  def __init__(self, weight=None, gamma=2., reduction='none'):
    PTnn.Module.__init__(self)
    self.weight = weight
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, input_tensor, target_tensor):
    log_prob = PTFunctional.log_softmax(input_tensor, dim=-1)
    prob = torch.exp(log_prob)
    loss = PTFunctional.nll_loss( ((1 - prob) ** self.gamma) * log_prob,
      target_tensor, weight=self.weight, reduction = self.reduction)
    loss = loss.mean()
    return loss

def calculateAccuracy(y_pred, y):
  top_pred = y_pred.argmax(1, keepdim = True)
  corrects = top_pred.eq(y.view_as(top_pred))
  acc = corrects.sum().float() / y.shape[0]
  return acc, corrects

def printNetworkInfo(network):
  for n, p in network.named_parameters():
    logging.info(p.device, '', n)

def checkCuda():
  print(f"PyTorch {torch.__version__}")
  print(f"Torchvision {torchvision.__version__}")
  if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))
    nbGpus = len(device_ids)
    print(f'CUDA {torch.version.cuda}')
    print(f'{nbGpus} GPU detected: {device_ids}')
    return 1
  else:
    DEVICE = torch.device("cpu")
    print('No CUDA, no GPU')
    return 0

def getDuration(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def setRandomSeed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def initializeLogging():
  logging.basicConfig(level=logging.INFO,
                      format='%(message)s',
                      handlers=[
                        logging.FileHandler(os.path.join(RSL_PATH,LOG_FILE),
mode='w'),
                        logging.StreamHandler(sys.stdout)])
