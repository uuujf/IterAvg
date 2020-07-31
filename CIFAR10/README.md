## CIFAR-10 experiments

The code generates part of Table 1 in the [paper](https://proceedings.icml.cc/static/paper_files/icml/2020/2773-Paper.pdf).

### Requirement
1. Python 3.6
2. PyTorch 1.3.1
3. TensorboardX

### Usage

#### Generate data
`python cifar2np.py`

#### VGG
- Open VGG folder: `cd ./VGG`
- Run SGD: `python sgd.py`
- Run iterate averaging: `python iteravg.py`


#### ResNet-18
- Open ResNet folder: `cd ./ResNet`
- Run SGD: `python sgd.py`
- Run iterate averaging: `python iteravg.py`


#### Hyperparameters
See the paper

### Acknowledgement
`swa.py` is borrowed from [pytorch/contrib](https://github.com/pytorch/contrib/tree/master/torchcontrib/optim) to fix Batch Normalization.