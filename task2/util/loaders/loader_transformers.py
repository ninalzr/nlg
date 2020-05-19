# Ignore slots for now.
# TODO: Figure out what to do with slots
# Remember to change load from file X, y
import os
import torch
import random
import torch.nn as nn
import numpy as np

from task2.util.lookup_transformers import Lookup
from functools import partial

def loader(data_folder, batch_size, lookup, src_lookup, tgt_lookup, min_seq_len_X=5, max_seq_len_X=1000, min_seq_len_y=5,
           max_seq_len_y=1000, MEI="", order = None):
    MEI = MEI.replace(" ", "_")
    pad_id = tgt_lookup.convert_tokens_to_ids(tgt_lookup.pad_token)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(data_folder, lookup, type="train",  min_seq_len_X=min_seq_len_X, max_seq_len_X=max_seq_len_X, min_seq_len_y=min_seq_len_y,
                  max_seq_len_y=max_seq_len_y, MEI=MEI, order = None),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, padding_idx=pad_id),
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        MyDataset(data_folder, lookup, type = "dev",  min_seq_len_X=min_seq_len_X, max_seq_len_X=max_seq_len_X, min_seq_len_y=min_seq_len_y,
                  max_seq_len_y=max_seq_len_y, MEI=MEI, order = None),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, padding_idx=pad_id))

    return train_loader, valid_loader


def paired_collate_fn(insts, padding_idx):
    # insts contains a batch_size number of (x, y) elements
    src_insts, tgt_insts = list(zip(*insts))

    src_max_len = max(len(inst) for inst in src_insts)  # determines max size for all examples

    src_seq_lengths = torch.tensor(list(map(len, src_insts)), dtype=torch.long)
    src_seq_tensor = torch.tensor(np.array([inst + [padding_idx] * (src_max_len - len(inst)) for inst in src_insts]),
                                  dtype=torch.long)
    src_seq_mask = torch.tensor(np.array([[1] * len(inst) + [0] * (src_max_len - len(inst)) for inst in src_insts]),
                                dtype=torch.long)

    src_seq_lengths, perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[perm_idx]
    src_seq_mask = src_seq_mask[perm_idx]
    tgt_max_len = max(len(inst) for inst in tgt_insts)

    tgt_seq_lengths = torch.tensor(list(map(len, tgt_insts)), dtype=torch.long)
    tgt_seq_tensor = torch.tensor(np.array([inst + [padding_idx] * (tgt_max_len - len(inst)) for inst in tgt_insts]),
                                  dtype=torch.long)
    tgt_seq_mask = torch.tensor(np.array([[1] * len(inst) + [0] * (tgt_max_len - len(inst)) for inst in tgt_insts]),
                                dtype=torch.long)

    tgt_seq_lengths = tgt_seq_lengths[perm_idx]
    tgt_seq_tensor = tgt_seq_tensor[perm_idx]
    tgt_seq_mask = tgt_seq_mask[perm_idx]

    return ((src_seq_tensor, src_seq_lengths, src_seq_mask), (tgt_seq_tensor, tgt_seq_lengths, tgt_seq_mask))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,  lookup, type, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, MEI, order = None):
        self.root_dir = root_dir

        self.X = []  # this will store joined sentences
        self.y = []  # this will store the output
        self.lookup = lookup


        if order == 'seq':
            print("Input sentences are joined in sequential order.")
            with open(os.path.join(root_dir, type, MEI + '_output_seq.txt'), 'r') as f:
                y = [self.lookup.encode(y.strip(), add_bos_eos_tokens=True) for y in f]
            with open(os.path.join(root_dir, type, MEI + '_sentences_seq.txt'), 'r') as g:
                X = [self.lookup.encode(x.strip(), add_bos_eos_tokens=True) for x in g]

        else:

            print("Input sentences are in non-sequential order")

            with open(os.path.join(root_dir, type, MEI + '_output.txt'), 'r') as f:
                y = [self.lookup.encode(y.strip(), add_bos_eos_tokens=True) for y in f]
            with open(os.path.join(root_dir, type, MEI + '_sentences.txt'), 'r') as g:
                X = [self.lookup.encode(x.strip(), add_bos_eos_tokens=True) for x in g]




        cut_over_X = 0
        cut_under_X = 0
        cut_over_y = 0
        cut_under_y = 0

        # max len
        for (sx, sy) in zip(X, y):
            if len(sx) > max_seq_len_X:
                cut_over_X += 1
            elif len(sx) < min_seq_len_X + 2:
                cut_under_X += 1
            elif len(sy) > max_seq_len_y:
                cut_over_y += 1
            elif len(sy) < min_seq_len_y + 2:
                cut_under_y += 1
            else:
                self.X.append(sx)
                self.y.append(sy)

        c = list(zip(self.X, self.y))
        random.shuffle(c)
        self.X, self.y = zip(*c)
        self.X = list(self.X)
        self.y = list(self.y)

        print("Dataset [{}] loaded with {} out of {} ({}%) instances.".format(type, len(self.X), len(X),
                                                                              float(100. * len(self.X) / len(X))))
        print("\t\t For X, {} are over max_len {} and {} are under min_len {}.".format(cut_over_X, max_seq_len_X,
                                                                                       cut_under_X, min_seq_len_X))
        print("\t\t For y, {} are over max_len {} and {} are under min_len {}.".format(cut_over_y, max_seq_len_y,
                                                                                       cut_under_y, min_seq_len_y))

        assert (len(self.X) == len(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]