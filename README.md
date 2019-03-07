
# Transformer

This is a pytorch implementation of the
[Transformer](https://arxiv.org/abs/1706.03762) model like
[tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor).

## Prerequisite

I tested it with PyTorch 1.0.0 and Python 3.6.8.

It's using [SpaCy](https://spacy.io/usage/) to tokenize languages for wmt32k
dataset. So, if you want to run `wmt32k` problem which is a de/en translation
dataset, you should download language models first with the following command.

```
$ pip install spacy
$ python -m spacy download en
$ python -m spacy download de
```

## Usage

1. Train a model.
```
$ python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data
or
$ python train.py --problem lm1b --output_dir ./output --data_dir ./lm1b_data
```

If you want to try `fast_transformer`, give a `model` argument after installing
[tcop-pytorch](https://github.com/tunz/tcop-pytorch).
```
$ python train.py --problem lm1b --output_dir ./output --data_dir ./lm1b_data --model fast_transformer
```


2. You can translate a single sentence with the trained model.
```
$ python decoder.py --translate --data_dir ./wmt32k_data --model_dir ./output/last/models
```
