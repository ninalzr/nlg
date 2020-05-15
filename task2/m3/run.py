# add package root
import os, sys, json
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
import numpy as np
import random

from task2.util.lookup_transformers import Lookup
from task2.util.utils import select_processing_device, clean_sequences


from task2.m3.model import EncoderDecoder
from task2.components.encoders.BERTEncoder import Encoder
from task2.components.decoders.BERTDecoder import Decoder



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = os.path.join("../..", "tiny")
src_lookup = Lookup('bert')
tgt_lookup = Lookup('bert')
lookup = Lookup('bert')
min_seq_len_X = 10
max_seq_len_X = 1000
min_seq_len_y = min_seq_len_X
max_seq_len_y = max_seq_len_X
hidden_size = 768
MEI = "Management Overview"
vocab_size = lookup.__len__()

encoder = Encoder(vocab_size=vocab_size, device=device)
decoder = Decoder(hidden_size=hidden_size, vocab_size=vocab_size, device=device)

model = EncoderDecoder(src_lookup=src_lookup, tgt_lookup=tgt_lookup,
                       encoder=encoder, decoder=decoder, device=device)

model.eval()
model.load_checkpoint(os.path.join("..", "train", "m3-"+MEI.replace(" ","_")), extension="last")


"""
joined_sentences = "In recent years, the company did not publish relevant ESG reports.A board committee at the company is responsible for overseeing governance issues only.The company lacks an environmental policy.Available evidence suggests the company does not have standards aimed at social supply chain issues.The social supply chain standard lacks direction on managing forced and child labour.The company's whistleblower programme has adequate measures."
x = torch.tensor(np.array( [ joined_sentences ] ), dtype=torch.long).to(device)
"""
