
## Why use Pytorch Distributed Data Parallelism (DDP)?
Pythorch offers different varieties of parallelism.

DistributedDataParallel is multi-process parallelism.
- can make use of multiple gpus on different machines.
- not easy to understand and debug

DataParallel is single-process multi-thread parallelism and thus works only one a single server
- less scalable and efficient since a single process performs the synchronization (update the model, combining loss and gradient).
- easier to implement and debug

One of the main issues with DDP is the lack of simple, easy to understand and complete enough documentation. So let's give it a try!

## The example and how to use it
the example consists in a basic classifier processing the MNIST dataset.
```bash
python ./main.py train -d $DATAPATH
python ./main.py test -d $DATAPATH -f $MODELFILE
```

## DDP recipe

Here are the key elements to successfully use DDP

#### Initialize the environment variables (main.py)
```python
# The master address (127.0.0.1 for local node)
os.environ['MASTER_ADDR'] = 182.168.1.101
# Communication port between gpus
os.environ['MASTER_PORT'] = 8765
# list of gpu ID on the local server
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
```

#### Spawn processes on all the local gpus (main.py)
- train_function = the function performing the train operations)
- args = a dictionary of args used by the train function
```python
torch.multiprocessing.spawn(train_function, nprocs=3, args=arg_dict)
```

train_function takes two parameters (classif.py)
- gpu : the rank of the gpu on the local server
- args: the above arguments

#### Initialize the  distributed process group (classif.py). <br>
This call block any operations until all the processes have joined
- worldSize = the number of gpus, all nodes considered
- rank = the rank of the gpu among all the node gpus
```python
torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://',
                                     world_size=worldSize,
                                     rank=rank)
```
#### Initialize the random generator (classif.py)<br>
so that everyone works on the same basis

```python
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  ```
#### Initialize the dataloader (dataloader.py)<br>
Declare data sampler and iterator
- worldSize = the number of gpus, all nodes considered
- rank = the rank of the gpu  among all the node gpus
- num_workers = the number of sub-processes to use for data loading
- pin_memory = True in order to speed up the host to device transfer of data

The important thing is to shuffle at the sampler level and not at the iterator one for the gpus to share the same order of data.

```python
sampler = torch.utils.data.distributed.DistributedSampler(
                     data,
                     shuffle=True,
                     num_replicas=worldSize,
                     rank=rank)
iterator = torch.utils.data.DataLoader(data,
                     shuffle=False,
                     batch_size=batchSize,
                     num_workers=NUM_WORKERS,
                     pin_memory=True,
                     sampler=sampler)
```

#### Distribute the model (classif.py)
```python
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
```

#### That's about all...
Or so! Don't forget to update the data sampler epoch otherwise the batches will be identical  across the epochs
```python
sampler.set_epoch(epoch)
```
## Documentation

Here are some of the links used to create this example

The pytorch tutos on DDP
- https://pytorch.org/tutorials/beginner/dist_overview.html
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://pytorch.org/tutorials/intermediate/dist_tuto.html

Some other tutorials
- this one is very good: https://github.com/bentrevett/pytorch-image-classification
- another one: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- and another one: https://towardsdatascience.com/how-to-scale-training-on-multiple-gpus-dae1041f49d2

About data synchronization
- https://gist.github.com/ResidentMario/dc542fc26a142a9dce85b258835c45ad

Saving and loading models with DDP
- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://github.com/pytorch/examples/blob/master/imagenet/main.py
