"""
 multi-gpu, multi-server distributed learning test using pytorch DDT
 test case: a basic classifier processing  MNIST
 author: georand
 source: https://github.com/georand/distributedpytorch
 date: 2021
"""

DEBUG = False
#DEBUG = True

# node addreses and gpu lists used for distributed learning
# the first node will  be the master node
# 2 nodes, 5 gpus
DDTNodes=[
  {'address':'172.16.12.8', 'gpus':'0,1'},
  {'address':'172.16.12.7', 'gpus':'0,1'},
]
# a single node, 2 gpus
#DDTNodes=[ {'address':'127.0.0.1', 'gpus':'0,1'} ]

# the master node ip and the port used for communication with the master node
MASTER_ADDR = DDTNodes[0]['address']
MASTER_PORT = '6779'

MODEL_NAME = 'resnet' # resnet | alexnet | vgg | squeezenet | densenet | inception

OPTIMIZER = 'adam' # adam | SGD

LOSS = 'cross_entropy' # cross_entropy | weighted_cross_entropy | focal_loss

DATA_PATH = './data'

RSL_PATH = './rsl'

LOG_FILE = 'test.log'

NB_EPOCHS = 2

BATCH_SIZE = 64*1

NUM_WORKERS = 2

SEED = 1234

# torchvision model : Flag for feature extracting.
# When False, finetune the whole model, when True, only update the reshaped layer params
FEATURE_EXTRACT = False

# torchvision model: use pretrained
USE_PRETRAINED = False

# if no CUDA number of thread used
NUM_THREADS = 32
