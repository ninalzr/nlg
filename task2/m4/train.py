import os, sys, json, random
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
from task2.util.train_transformers import train_t5
from task2.util.lookup_transformers import Lookup
from task2.util.loaders.loader_transformers import loader
from task2.components.encodersdecoders.T5 import T5

if __name__ == "__main__":
    batch_size = 4
    data_folder = os.path.join("../..", "tiny")
    src_lookup = Lookup('t5')
    tgt_lookup = Lookup('t5')
    lookup = Lookup('t5')
    min_seq_len_X = 10
    max_seq_len_X = 1000
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X
    MEI = "Management Overview"
    vocab_size = lookup.__len__()
    print(vocab_size)
    #order = None
    order = 'seq' #input sentences are in sequential order

    train_loader, valid_loader = loader(data_folder, batch_size, lookup, src_lookup, tgt_lookup, min_seq_len_X,
                                        max_seq_len_X,  min_seq_len_y, max_seq_len_y, MEI=MEI, order = None)

    print("Loading done, train instances {}, dev instances {}, vocab size src/tgt {}/{}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(src_lookup), len(tgt_lookup)))

    hidden_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5(hidden_size= hidden_size, vocab_size=vocab_size, device=device)


    model_store_path = os.path.join("..", "train", "m4-" + MEI.replace(" ", "_"))
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
    criterion = nn.NLLLoss()
    lr = None
    max_epochs = 50

    train_t5(model,
              train_loader,
              valid_loader,
              tgt_lookup,
              optimizer=optimizer,
              criterion=criterion,
              max_epochs = max_epochs,
              lr = lr,
              model_store_path = model_store_path)



