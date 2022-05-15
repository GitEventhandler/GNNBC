# Graph Neural Networks Beyond Compromise Between Attribute and Topology

This repository contains a PyTorch implementation of ["Graph Neural Networks Beyond Compromise Between Attribute and Topology"](https://yangliang.github.io/pdf/www22.pdf).

## Runtime Environment

* python 3.8
* numpy 1.20.3
* pytorch 1.8.0
* pytorch-sparse 0.6.11

## Dataset Source

All datasets are downloaded from package torch_geometric and saved as series of .pt file without any preprocess procedure. You can download the zipped dataset from release page of this repo and extract them to "%PROJECT_ROOT%/dataset" folder.

## Run All Benchmarks

```
./train.sh
```

## Citation

```
@article{yangWWW2022gnnbc,
  title = {Graph Neural Networks Beyond Compromise Between Attribute and Topology},
  author = {Liang Yang, Wenmiao Zhou and Weihang Peng, Bingxin Niu and Junhua Gu, Chuan Wang and Xiaochun Cao, Dongxiao He},
  year = {2022},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022},
}
```

