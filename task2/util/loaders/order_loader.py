import os, sys, json, random
sys.path.append("../../..")

from task2.util.lookup import Lookup
import numpy as np
import torch
import torch.utils.data

def loader(data_folder, batch_size, MEI = ""):
    MEI = MEI.replace(" ","_")
    
    train_loader = torch.utils.data.DataLoader(
        MyDataset(data_folder, "train", MEI),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        MyDataset(data_folder, "dev", MEI),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=paired_collate_fn)
    
    return train_loader, valid_loader

def paired_collate_fn(insts):
    def _translate_order_to_index(inst):
        # input is ['1','','2','','3','5','4','']
        # index is  #1  #2 #3  #4 #5  #6  #7  #8
        # output will become [1,3,5,7,6,0,0,0] (the indexes of the sentences)
        output = []
        for i in range(1, len(inst)+1):            
            if str(i) in inst:
                index = inst.index(str(i))+1
                output.append(index)
        output += [0]*(len(inst)-len(output))
        return output
        
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts = list(zip(*insts))
    # x is (encoded) slots, y is order
    # x is a list of all slots [1,2,4,3,0,15,2,3], order is a ['1','','2','3','5','4',''] list
    
    max_len = len(tgt_insts[0]) # determines max size of order (# of slot_groups)
    
    X = torch.tensor( src_insts , dtype=torch.long)
     
    # for y, process all elements into padded arrays   
    ys = [ _translate_order_to_index(inst) for inst in tgt_insts ]
    y = torch.tensor( ys , dtype=torch.long)
    
    return (X, y)    

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type, MEI):  
        self.root_dir = root_dir

        self.slots_object = torch.load(os.path.join(root_dir,MEI+"_slots_object.pt"))
    
        self.X = torch.load(os.path.join(root_dir,MEI.replace(" ","_")+"_slots_"+type+".pt"))
        self.y = torch.load(os.path.join(root_dir,MEI.replace(" ","_")+"_order_"+type+".pt"))
      
        c = list(zip(self.X, self.y))
        random.shuffle(c)
        self.X, self.y = zip(*c)
        self.X = list(self.X)
        self.y = list(self.y)
                    
        print("Dataset [{}] loaded with {} instances.".format(type, len(self.X)) )
        
        assert(len(self.X)==len(self.y))
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]

"""
MEI = "Business Ethics"
batch_size = 4
min_seq_len_X = 0
max_seq_len_X = 1000
min_seq_len_y = min_seq_len_X
max_seq_len_y = max_seq_len_X    

data_folder = os.path.join("..", "..", "data", "ready")
src_lookup_prefix = os.path.join("..", "..", "data", "lookup", "bpe", MEI.replace(" ","_"))
tgt_lookup_prefix = os.path.join("..", "..", "data", "lookup", "bpe", MEI.replace(" ","_")) #os.path.join("task2", "data", "lookup", "bpe", MEI.replace(" ","_"))
src_lookup = Lookup(type="bpe")
tgt_lookup = Lookup(type="bpe")

src_lookup.load(src_lookup_prefix)    
tgt_lookup.load(tgt_lookup_prefix)    
train_loader, valid_loader = loader(data_folder, batch_size, src_lookup, tgt_lookup, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, MEI = MEI)


print("Loading done, train instances {}, dev instances {}, vocab size src/tgt {}/{}\n".format(
    len(train_loader.dataset.slots),
    len(valid_loader.dataset.slots),        
    len(src_lookup), len(tgt_lookup)))

s = train_loader.dataset.slots_object["slots"]
v = train_loader.dataset.slots_object["values"]
x = train_loader.dataset.X
i = train_loader.dataset.y
slots = train_loader.dataset.slots

slot_sizes = []
for i in range(len(s)):
    slot_name = s[i]
    slot_sizes.append(len(v[slot_name])) # 0 is for not found or FAILED
    print("\t Slot [{}] has [{}] values".format(slot_name, slot_sizes[-1]))
"""