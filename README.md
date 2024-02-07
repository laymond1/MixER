# MixER: Mixup-based Experience Replay for Online Class-Incremental Learning

Official Code for ["MixER: Mixup-based Experience Replay for Online Class-Incremental Learning"](). 
Our code is based on the implementation of [Mammoth](https://github.com/aimagelab/mammoth).

## Running Experiments

Examples:

CIFAR-10
  ```
  python main.py --dataset seq-cifar10 --model MixER --buffer_size 500 --load_best_args
  ```
CIFAR-100
  ```
  python main.py --dataset seq-cifar100 --model MixER --buffer_size 1000 --load_best_args
  ```
Tiny-ImageNet
  ```
  python main.py --dataset seq-tinyimg --model MixER --buffer_size 2000 --load_best_args
  ```

## Cite Our Work
If you find this code useful, please consider reference in our paper:

```
@InProceedings{lim2024mixer,
    author    = {Lim, Won-Seon and Yu Zhou and Kim, Dae-Won and Lee, Jaesung},
    title     = {MixER: Mixup-based Experience Replay for Online Class-Incremental Learning},
    journal   = {IEEE ACCESS},
    volume    = {},
    number    = {},
    year      = {2024},
    publisher = {}
}
```
