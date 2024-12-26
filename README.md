# MixER: Mixup-based Experience Replay for Online Class-Incremental Learning

Official Code for ["MixER: Mixup-based Experience Replay for Online Class-Incremental Learning"](). 
Our code is based on the implementation of [Mammoth](https://github.com/aimagelab/mammoth).

## Running Experiments

Examples:

CIFAR-10
  ```
  python main.py --dataset seq-cifar10 --model mixer --buffer_size 500 --lr 0.01 --batch_size 10 --minibatch_size 10 --n_epochs 1 --gamma 5.0
  ```
CIFAR-100
  ```
  python main.py --dataset seq-cifar100 --model mixer --buffer_size 1000 --lr 0.01 --batch_size 10 --minibatch_size 10 --n_epochs 1 --gamma 5.0
  ```
Tiny-ImageNet
  ```
  python main.py --dataset seq-tinyimg --model mixer --buffer_size 2000 --lr 0.01 --batch_size 10 --minibatch_size 10 --n_epochs 1 --gamma 5.0
  ```

## Cite Our Work
If you find this code useful, please consider reference in our paper:

```
@InProceedings{lim2024mixer,
    author    = {Lim, Won-Seon and Yu Zhou and Kim, Dae-Won and Lee, Jaesung},
    title     = {MixER: Mixup-based Experience Replay for Online Class-Incremental Learning},
    journal   = {IEEE ACCESS},
    volume    = {12},
    pages    = {41801--41814},
    year      = {2024},
    publisher = {IEEE}
}
```
