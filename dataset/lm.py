from collections import Counter, OrderedDict
import glob
import io
import os
import pickle

import torch
from torchtext import data
from tqdm import tqdm

from dataset import common

# pylint: disable=arguments-differ


def split_tokenizer(x):
    return x.split()


def read_examples(paths, fields, data_dir, mode, filter_pred, num_shard):
    data_path_fmt = data_dir + '/examples-' + mode + '-{}.pt'
    data_paths = [data_path_fmt.format(i) for i in range(num_shard)]
    writers = [open(data_path, 'wb') for data_path in data_paths]
    shard = 0

    for path in paths:
        print("Preprocessing {}".format(path))

        with io.open(path, mode='r', encoding='utf-8') as trg_file:
            for trg_line in tqdm(trg_file, ascii=True):
                trg_line = trg_line.strip()
                if trg_line == '':
                    continue

                example = data.Example.fromlist([trg_line], fields)
                if not filter_pred(example):
                    continue

                pickle.dump(example, writers[shard])
                shard = (shard + 1) % num_shard

    for writer in writers:
        writer.close()

    # Reload pickled objects, and save them again as a list.
    common.pickles_to_torch(data_paths)

    examples = torch.load(data_paths[0])
    return examples, data_paths


class LM1b(data.Dataset):
    urls = ["http://www.statmt.org/lm-benchmark/"
            "1-billion-word-language-modeling-benchmark-r13output.tar.gz"]
    name = 'lm1b'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.trg)

    @classmethod
    def splits(cls, fields, data_dir, root='.data', **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('trg', fields[0])]

        filter_pred = kwargs['filter_pred']

        expected_dir = os.path.join(root, cls.name)
        path = (expected_dir if os.path.exists(expected_dir)
                else cls.download(root))

        lm_data_dir = "1-billion-word-language-modeling-benchmark-r13output"

        train_files = [
            os.path.join(path,
                         lm_data_dir,
                         "training-monolingual.tokenized.shuffled",
                         "news.en-%05d-of-00100" % i) for i in range(1, 100)
        ]
        train_examples, data_paths = \
            read_examples(train_files, fields, data_dir, 'train',
                          filter_pred, 100)

        val_files = [
            os.path.join(path,
                         lm_data_dir,
                         "heldout-monolingual.tokenized.shuffled",
                         "news.en.heldout-00000-of-00050")
        ]
        val_examples, _ = read_examples(val_files, fields, data_dir,
                                        'val', filter_pred, 1)

        train_data = cls(train_examples, fields, **kwargs)
        val_data = cls(val_examples, fields, **kwargs)
        return (train_data, val_data, data_paths)


def len_of_example(example):
    return len(example.trg) + 1


def build_vocabs(trg_field, data_paths):
    trg_counter = Counter()
    for data_path in tqdm(data_paths, ascii=True):
        examples = torch.load(data_path)
        for x in examples:
            trg_counter.update(x.trg)

    specials = list(OrderedDict.fromkeys(
        tok for tok in [trg_field.unk_token,
                        trg_field.pad_token,
                        trg_field.init_token,
                        trg_field.eos_token]
        if tok is not None))
    trg_field.vocab = trg_field.vocab_cls(trg_counter, specials=specials,
                                          min_freq=300)


def prepare(max_length, batch_size, device, opt, data_dir):
    pad = '<pad>'
    load_preprocessed = os.path.exists(data_dir + '/target.pt')

    def filter_pred(x):
        return len(x.trg) < max_length

    if load_preprocessed:
        print("Loading preprocessed data...")
        trg_field = torch.load(data_dir + '/target.pt')['field']

        data_paths = glob.glob(data_dir + '/examples-train-*.pt')
        examples_train = torch.load(data_paths[0])
        examples_val = torch.load(data_dir + '/examples-val-0.pt')

        fields = [('trg', trg_field)]
        train = LM1b(examples_train, fields, filter_pred=filter_pred)
        val = LM1b(examples_val, fields, filter_pred=filter_pred)
    else:
        trg_field = data.Field(tokenize=split_tokenizer, batch_first=True,
                               is_target=True,
                               pad_token=pad, lower=True, eos_token='<eos>')

        print("Loading data... (this may take a while)")
        train, val, data_paths = \
            LM1b.splits(fields=(trg_field,),
                        data_dir=data_dir,
                        filter_pred=filter_pred)
        # fields = [('trg', trg_field)]
        # data_paths = glob.glob(data_dir + '/examples-train-*.pt')
        # examples_train = torch.load(data_paths[0])
        # examples_val = torch.load(data_dir + '/examples-val-0.pt')
        # train = LM1b(examples_train, fields, filter_pred=filter_pred)
        # val = LM1b(examples_val, fields, filter_pred=filter_pred)

        print("Building vocabs... (this may take a while)")
        build_vocabs(trg_field, data_paths)

    print("Creating iterators...")
    train_iter, val_iter = common.BucketByLengthIterator.splits(
        (train, val),
        data_paths=data_paths,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        example_length_fn=len_of_example)

    opt.src_vocab_size = None
    opt.trg_vocab_size = len(trg_field.vocab)
    opt.src_pad_idx = None
    opt.trg_pad_idx = trg_field.vocab.stoi[pad]
    opt.has_inputs = False

    if not load_preprocessed:
        torch.save({'pad_idx': opt.trg_pad_idx, 'field': trg_field},
                   data_dir + '/target.pt')

    return train_iter, val_iter, opt
