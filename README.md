# FedED

Empty Classes Matter in Federated Learning under Label Distribution Skews

## Requirements
### Installation
Create a conda environment and install dependencies:
```python
conda create -n feded python=3.11.5
conda activate feded

pip install -r environment.txt

```
### Dataset
Here we provide the implementation on MNIST, Cifar10, Cifar100, and TinyImagenet datasets. The four datasets will be automatically downloaded in your datadir.

### Model Structure
As for the models used in the paper, we use a DNN with three fully connected layers for the MNIST dataset and the same model structure of Mobilenet_v2 as [CCVR](https://arxiv.org/pdf/2106.05001) for other datasets.

### Parameters
| Parameter        | Description                                                                                           |
|------------------|-------------------------------------------------------------------------------------------------------|
| `m`              | The model architecture. Options: `dnn`, `mobilenetv2`.                                                |
| `lbs`            | Local batch size.                                                                                     |
| `lr`             | Learning rate.                                                                                        |
| `nc`             | Number of clients.                                                                                    |
| `jr`             | The client joining rate.                                                                              |
| `nb`             | Number of classes.                                                                                    |
| `data`           | Dataset to use. Options: `mnist `, `cifar10 `, `cifar100`, `tinyimagenet`.                            |
| `algo`           | Algorithm to use.                                                                                     |
| `gr`             | Global_rounds.                                                                                        |
| `did`            | Device id.                                                                                            |
| `partition`      | The data partitioning strategy.                                                                       |
| `al`             | The Dirichlet distribution coefficient.                                                               |
| `ls`             | local_epochs.                                                                                         |
| `lam `           | The coefficient for empty class distillation loss in our method.                                      |

### Usage
Here is an example to run FedED on CIFAR10 with mobilenetv2:
```
python -u main.py -lbs 64 -nc 10 -jr 1 -nb 10 -data cifar10 -m mobilenetv2 -algo FedED -gr 100 -did 0 -aug True -lr 0.01 -partition noniid -al 0.05 -ls 10 -ed 1e-5 -lam 0.1
```

### Acknowledgement
We borrow some codes from [CCVR](https://arxiv.org/pdf/2106.05001) and [PFLlib](https://github.com/TsingZ0/PFLlib).






