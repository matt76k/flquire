# FLQuire

## Overview

FLQuire is a float-like qurie. This repository includes flquire emulator for [resnet9](https://github.com/matthias-wright/cifar10-resnet)

## Requirement

* [PyTorch (1.9.0)](https://pytorch.org/)
* [NumPy (1.21.2)](http://www.numpy.org/)
* [Namagiri](https://github.com/matt76k/namagiri)

## Usage

### Install Namagiri

    git clone git@github.com:matt76k/namagiri.git
    cd namagiri
    cargo build --release
    cp target/release/libnamagiri.dylib ../namagiri.so

### Test FLQuire

    python fl.py

### Training

you can retrain the model.

    python train.py

## Reference

* [resnet9](https://github.com/matthias-wright/cifar10-resnet)
* [fuse](https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/fx/experimental/fuser.py)
* [im2col](https://qiita.com/kuroitu/items/35d7b5a4bde470f69570)
  
## License
This project is licensed under the [MIT Licence](https://choosealicense.com/licenses/mit/)