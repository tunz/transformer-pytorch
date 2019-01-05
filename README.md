
# Transformer

This is a pytorch implementation of the
[Transformer](https://arxiv.org/abs/1706.03762) model like
[tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor).

## Usage

1. Train a model.
```
python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data
```

2. You can translate a single sentence with the trained model.
```
python translate.py --data_dir ./wmt32k_data --model_dir ./output/last/models
```
