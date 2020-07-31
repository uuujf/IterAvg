## MNIST experiments

The code generates Figure 4 and Figure 5 in the [paper](https://proceedings.icml.cc/static/paper_files/icml/2020/2773-Paper.pdf).

### Requirement
1. Python 3.6
2. PyTorch 1.3.1

### Usage

#### Generate data
`python mnist2np.py`

#### Linear regression
Please adjust hyperparameter `bs` to control batch size.
- GD/SGD: `python sgd-linreg`
- PGD/PSGD: `python psgd-linreg`
- NGD/NSGD: `python nsgd-linreg`
- Plotting: `python plot.py`

#### Logistic regression
Please adjust hyperparameter `bs` to control batch size.
- GD/SGD: `python sgd-logreg`
- PGD/PSGD: `python psgd-logreg`
- NGD/NSGD: `python nsgd-logreg`
- Plotting: `python plot.py`

#### Hyperparameters
See the paper.