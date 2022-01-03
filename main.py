"""
 multi-gpu, multi-server distributed learning test using pytorch DDT
 test case: a basic classifier processing  MNIST
 author: georand
 source: https://github.com/georand/distributedpytorch
 date: 2021
"""

import sys
import os
import argparse
import socket, array, struct, fcntl # for ip address retrieval

import torch.multiprocessing as PTmp

from config import *
import utils
import classif

def getArgs():

  # common arguments
  parserCommon = argparse.ArgumentParser(add_help=False, )

  parserCommon.add_argument('--debug',
                     action='store_true', dest='debug', default=DEBUG,
                     help = 'debug mode')
  parserCommon.add_argument('-d', '--data_path', metavar='data_path',
                            action='store',type=str, dest='dataPath',
                            default=None, required=True, help = 'data path')
  parserCommon.add_argument('-b', '--batchSize', metavar='N', type=int,
                            dest='batchSize', default=BATCH_SIZE,
                            help = f'batch size (default: {BATCH_SIZE})')

  # main parser
  parser = argparse.ArgumentParser(description='Et aïe!')
  subparsers = parser.add_subparsers(dest='action',
                                     help='action to execute', required= True)

  # train arguments
  parserTrain = subparsers.add_parser('train',parents=[parserCommon],
                                      help="train model")
  parserTrain.add_argument('-e', '--epochs', metavar='N', type=int,
                            dest='nbEpochs', default=NB_EPOCHS,
                          help = f'number of training epochs (default: {NB_EPOCHS})')
  parserTrain.add_argument('-f', '--file', metavar='file_path', action='store',
                            type=str, dest='checkpointFile', default=None,
                            help = 'training checkpoint file')

  # test armuments
  parserTest = subparsers.add_parser('test',parents=[parserCommon],
                                     help="test model")
  parserTest.add_argument('-f', '--file', metavar='file_path', action='store',
                           type=str, dest='checkpointFile', default=None, required=True,
                           help = 'model file')


  return parser.parse_args()

def getLocalInterfaces():
  """
  Returns a dictionary of name:ip key value pairs.
  source code from https://gist.github.com/pklaus/289646
  """
  MAX_BYTES = 4096
  FILL_CHAR = b'\0'
  SIOCGIFCONF = 0x8912
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  names = array.array('B', MAX_BYTES * FILL_CHAR)
  names_address, names_length = names.buffer_info()
  mutable_byte_buffer = struct.pack('iL', MAX_BYTES, names_address)
  mutated_byte_buffer = fcntl.ioctl(sock.fileno(), SIOCGIFCONF, mutable_byte_buffer)
  max_bytes_out, names_address_out = struct.unpack('iL', mutated_byte_buffer)
  namestr = names.tobytes()
  namestr[:max_bytes_out]
  bytes_out = namestr[:max_bytes_out]
  ip_dict = {}
  for i in range(0, max_bytes_out, 40):
    name = namestr[ i: i+16 ].split(FILL_CHAR, 1)[0]
    name = name.decode('utf-8')
    ip_bytes   = namestr[i+20:i+24]
    full_addr = []
    for netaddr in ip_bytes:
      if isinstance(netaddr, int):
        full_addr.append(str(netaddr))
      elif isinstance(netaddr, str):
        full_addr.append(str(ord(netaddr)))
    ip_dict[name] = '.'.join(full_addr)

  return ip_dict

def getDDTInfo():
  '''
   get active gpu list for present node
   firstLocalRank:rank of the first gpu within all the nodes
   worldSize:
  '''
  ip = getLocalInterfaces()
  node = None
  worldSize = 0
  firstLocalRank = 0
  for n in DDTNodes:
    if n['address'] in ip.values():
      node = n
    nbGpus = len(n['gpus'].split(','))
    if  not node:
      firstLocalRank += nbGpus
    worldSize += nbGpus

  return node['gpus'], firstLocalRank, worldSize

if __name__ == "__main__":

  args = getArgs()
  DEBUG = args.debug
  args.gpus, args.firstLocalRank, args.worldSize = getDDTInfo()
  utils.initializeLogging()

  if not utils.checkCuda():
    cuda = False
    torch.set_num_threads(NUM_THREADS)
  else:
    cuda = True

  print('========================= start =========================')

  if cuda:
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    #os.environ['NCCL_IB_DISABLE'] = '1'
    if args.action == 'train':
      PTmp.spawn(classif.train, nprocs=len(args.gpus.split(',')), args=(args,))
    elif args.action == 'test':
      PTmp.spawn(classif.test, nprocs=len(args.gpus.split(',')), args=(args,))
  else: # no CUDA => set gpuID to -1
    if args.action == 'train':
      train(-1, args)
    elif args.action == 'test':
      test(-1, args)

  print('========================= end ==========================')
