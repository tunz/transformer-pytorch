from collections import Counter, OrderedDict
import glob
import io
import os
import pickle
import re

import torch
from torchtext import data
import spacy
from tqdm import tqdm

from dataset import common

# pylint: disable=arguments-differ

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


def read_examples(paths, exts, fields, data_dir, mode, filter_pred, num_shard):
    data_path_fmt = data_dir + '/examples-' + mode + '-{}.pt'
    data_paths = [data_path_fmt.format(i) for i in range(num_shard)]
    writers = [open(data_path, 'wb') for data_path in data_paths]
    shard = 0

    for path in paths:
        print("Preprocessing {}".format(path))
        src_path, trg_path = tuple(path + x for x in exts)

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in tqdm(zip(src_file, trg_file),
                                           ascii=True):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line == '' or trg_line == '':
                    continue

                example = data.Example.fromlist(
                    [src_line, trg_line], fields)
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


class WMT32k(data.Dataset):
    urls = ['http://data.statmt.org/wmt18/translation-task/'
            'training-parallel-nc-v13.tgz',
            'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
            'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
            'http://data.statmt.org/wmt17/translation-task/dev.tgz']
    name = 'wmt32k'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    @classmethod
    def splits(cls, exts, fields, data_dir, root='.data', **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        filter_pred = kwargs['filter_pred']

        expected_dir = os.path.join(root, cls.name)
        path = (expected_dir if os.path.exists(expected_dir)
                else cls.download(root))

        train_files = ['training-parallel-nc-v13/news-commentary-v13.de-en',
                       'commoncrawl.de-en',
                       'training/europarl-v7.de-en']
        train_files = map(lambda x: os.path.join(path, x), train_files)
        train_examples, data_paths = \
            read_examples(train_files, exts, fields, data_dir, 'train',
                          filter_pred, 100)

        val_files = [os.path.join(path, 'dev/newstest2013')]
        val_examples, _ = read_examples(val_files, exts, fields, data_dir,
                                        'val', filter_pred, 1)

        train_data = cls(train_examples, fields, **kwargs)
        val_data = cls(val_examples, fields, **kwargs)
        return (train_data, val_data, data_paths)


def len_of_example(example):
    return max(len(example.src) + 1, len(example.trg) + 1)


def build_vocabs(src_field, trg_field, data_paths):
    src_counter = Counter()
    trg_counter = Counter()
    for data_path in tqdm(data_paths, ascii=True):
        examples = torch.load(data_path)
        for x in examples:
            src_counter.update(x.src)
            trg_counter.update(x.trg)

    specials = list(OrderedDict.fromkeys(
        tok for tok in [src_field.unk_token,
                        src_field.pad_token,
                        src_field.init_token,
                        src_field.eos_token]
        if tok is not None))
    src_field.vocab = src_field.vocab_cls(src_counter, specials=specials,
                                          min_freq=50)
    trg_field.vocab = trg_field.vocab_cls(trg_counter, specials=specials,
                                          min_freq=50)


def prepare(max_length, batch_size, device, opt, data_dir):
    pad = '<pad>'
    load_preprocessed = os.path.exists(data_dir + '/source.pt')

    def filter_pred(x):
        return len(x.src) < max_length and len(x.trg) < max_length

    if load_preprocessed:
        print("Loading preprocessed data...")
        src_field = torch.load(data_dir + '/source.pt')['field']
        trg_field = torch.load(data_dir + '/target.pt')['field']

        data_paths = glob.glob(data_dir + '/examples-train-*.pt')
        examples_train = torch.load(data_paths[0])
        examples_val = torch.load(data_dir + '/examples-val-0.pt')

        fields = [('src', src_field), ('trg', trg_field)]
        train = WMT32k(examples_train, fields, filter_pred=filter_pred)
        val = WMT32k(examples_val, fields, filter_pred=filter_pred)
    else:
        src_field = data.Field(tokenize=tokenize_de, batch_first=True,
                               pad_token=pad, lower=True, eos_token='<eos>')
        trg_field = data.Field(tokenize=tokenize_en, is_target=True,
                               batch_first=True,
                               pad_token=pad, lower=True, eos_token='<eos>')

        print("Loading data... (this may take a while)")
        train, val, data_paths = \
            WMT32k.splits(exts=('.de', '.en'),
                          fields=(src_field, trg_field),
                          data_dir=data_dir,
                          filter_pred=filter_pred)

        print("Building vocabs... (this may take a while)")
        build_vocabs(src_field, trg_field, data_paths)

    print("Creating iterators...")
    train_iter, val_iter = common.BucketByLengthIterator.splits(
        (train, val),
        data_paths=data_paths,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        example_length_fn=len_of_example)

    opt.src_vocab_size = len(src_field.vocab)
    opt.trg_vocab_size = len(trg_field.vocab)
    opt.src_pad_idx = src_field.vocab.stoi[pad]
    opt.trg_pad_idx = trg_field.vocab.stoi[pad]

    if not load_preprocessed:
        torch.save({'pad_idx': opt.src_pad_idx, 'field': src_field},
                   data_dir + '/source.pt')
        torch.save({'pad_idx': opt.trg_pad_idx, 'field': trg_field},
                   data_dir + '/target.pt')

    return train_iter, val_iter, opt
