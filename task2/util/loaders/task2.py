import os, sys, json, random
sys.path.append("../../..")

from task2.util.lookup import Lookup
import numpy as np
import torch
import torch.utils.data
from functools import partial

"""
This loader creates X as batches of (sentences_joined, lengths, masks, slots) 
and y as (output, lengths, masks)
"""


def loader(data_folder, batch_size, src_lookup, tgt_lookup, min_seq_len_X = 5, max_seq_len_X = 1000, min_seq_len_y = 5,
           max_seq_len_y = 1000, MEI = ""):
    MEI = MEI.replace(" ","_")
    pad_id = tgt_lookup.convert_tokens_to_ids(tgt_lookup.pad_token)
    
    train_loader = torch.utils.data.DataLoader(
        MyDataset(data_folder, "train", min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, MEI),
        num_workers=0,
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, padding_idx = pad_id),
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        MyDataset(data_folder, "dev", min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, MEI),
        num_workers=0,
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, padding_idx = pad_id))
    
    return train_loader, valid_loader

def paired_collate_fn(insts, padding_idx):
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts, slot_insts = list(zip(*insts))
   
    # now src is a batch_size(=64) array of x0 .. x63, and tgt is y0 .. x63 ; xi is variable length
    # ex: if a = [(1,2), (3,4), (5,6)]
    # then b, c = list(zip(*a)) => b = (1,3,5) and b = (2,4,6)
    
    # src_insts is now a tuple of batch_size Xes (x0, x63) where xi is an instance
    #src_insts, src_lenghts, tgt_insts, tgt_lenghts = length_collate_fn(src_insts, tgt_insts)       
    
    src_max_len = max(len(inst) for inst in src_insts) # determines max size for all examples
    
    src_seq_lengths = torch.tensor(list(map(len, src_insts)), dtype=torch.long)    
    src_seq_tensor = torch.tensor(np.array( [ inst + [padding_idx] * (src_max_len - len(inst)) for inst in src_insts ] ), dtype=torch.long)
    src_seq_mask = torch.tensor(np.array( [ [1] * len(inst) + [0] * (src_max_len - len(inst)) for inst in src_insts ] ), dtype=torch.long)
    
    src_seq_lengths, perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[perm_idx]   
    src_seq_mask = src_seq_mask[perm_idx]
    tgt_max_len = max(len(inst) for inst in tgt_insts)
    
    tgt_seq_lengths = torch.tensor(list(map(len, tgt_insts)), dtype=torch.long)    
    tgt_seq_tensor = torch.tensor(np.array( [ inst + [padding_idx] * (tgt_max_len - len(inst)) for inst in tgt_insts ] ), dtype=torch.long)
    tgt_seq_mask = torch.tensor(np.array( [ [1] * len(inst) + [0] * (tgt_max_len - len(inst)) for inst in tgt_insts ] ), dtype=torch.long)
    src_slots = torch.tensor(slot_insts, dtype=torch.long)
    
    tgt_seq_lengths = tgt_seq_lengths[perm_idx]
    tgt_seq_tensor = tgt_seq_tensor[perm_idx]      
    tgt_seq_mask = tgt_seq_mask[perm_idx]   
    src_slots = src_slots[perm_idx]   
      
    return ((src_seq_tensor, src_seq_lengths, src_seq_mask, src_slots), (tgt_seq_tensor, tgt_seq_lengths, tgt_seq_mask))    

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, MEI):  
        self.root_dir = root_dir

        self.X = [] # this will store joined sentences
        self.y = [] # this will store the output
        self.slots = [] # will store the encoded slots
        self.slots_object = torch.load(os.path.join(root_dir,MEI+"_slots_object.pt"))
    
        X = torch.load(os.path.join(root_dir,MEI+"_sentences_joined_"+type+".pt"))
        y = torch.load(os.path.join(root_dir,MEI+"_output_"+type+".pt"))
        slots = torch.load(os.path.join(root_dir,MEI+"_slots_"+type+".pt"))
        
        cut_over_X = 0
        cut_under_X = 0
        cut_over_y = 0
        cut_under_y = 0
        
        # max len
        for (sx, sy, sl) in zip(X, y, slots):
            if len(sx) > max_seq_len_X:
                cut_over_X += 1
            elif len(sx) < min_seq_len_X+2:                
                cut_under_X += 1
            elif len(sy) > max_seq_len_y:
                cut_over_y += 1
            elif len(sy) < min_seq_len_y+2:                
                cut_under_y += 1
            else:
                self.X.append(sx)
                self.y.append(sy)         
                self.slots.append(sl)

        c = list(zip(self.X, self.y, self.slots))
        random.shuffle(c)
        self.X, self.y, self.slots = zip(*c)
        self.X = list(self.X)
        self.y = list(self.y)
        self.slots = list(self.slots)
                    
        print("Dataset [{}] loaded with {} out of {} ({}%) instances.".format(type, len(self.X), len(X), float(100.*len(self.X)/len(X)) ) )
        print("\t\t For X, {} are over max_len {} and {} are under min_len {}.".format(cut_over_X, max_seq_len_X, cut_under_X, min_seq_len_X))
        print("\t\t For y, {} are over max_len {} and {} are under min_len {}.".format(cut_over_y, max_seq_len_y, cut_under_y, min_seq_len_y))
        
        assert(len(self.X)==len(self.y))
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx], self.slots[idx]