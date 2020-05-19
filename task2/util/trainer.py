import sys
sys.path.insert(0, '../..')

import os, subprocess, gc, time
from collections import OrderedDict
import torch
import torch.nn as nn
from task2.util.log import Log
from tqdm import tqdm
import numpy as np
from task2.util.validation_metrics import evaluate
from task2.util.utils import pretty_time, clean_sequences
from task2.util.lookup import Lookup

def _plot_pgen_activations(X, y, src_lookup, tgt_lookup, decoder_dict, epoch, log_object):
    input_labels = []
    X_list = X[0][0].cpu().tolist() 
      
    for id in X_list:
        input_labels.append(src_lookup.convert_ids_to_tokens(int(id)))
    
    # y tgt labels for the first example    
    y_list = y[0][0].cpu().tolist()
    output_labels = []
    for id in y_list:
        output_labels.append(tgt_lookup.convert_ids_to_tokens(int(id)))    
    
    # map weights
    tdata = decoder_dict["p_gen"] # [batch_size, dec_seq_len-1]
    tdata = tdata[0:1,:].cpu().tolist()
    log_object.plot_heatmap(tdata, input_labels=["copy"], output_labels=output_labels, epoch=epoch, file_prefix="p_gen")

def _plot_copy_activations(X, y, src_lookup, tgt_lookup, decoder_dict, epoch, log_object):
    input_labels = []
    X_list = X[0].cpu().tolist()    
    for id in X_list:
        input_labels.append(src_lookup.convert_ids_to_tokens(int(id)))
    
    # y tgt labels for the first example    
    y_list = y[0].cpu().tolist()
    output_labels = []
    for id in y_list:
        output_labels.append(tgt_lookup.convert_ids_to_tokens(int(id)))    
    
    # map weights
    tdata = decoder_dict["copy_activation"] # [batch_size, dec_seq_len-1]
    tdata = tdata[0,:].cpu().tolist()
    data = np.zeros((1, len(tdata)))
    for i in range(len(tdata)):
        if tdata[i]==True:
            data[0,i]=1.

    log_object.plot_heatmap(data, input_labels=["copy"], output_labels=output_labels, epoch=epoch, file_prefix="copy")

def _plot_attention_weights(X, y, src_lookup, tgt_lookup, attention_weights, epoch, log_object):
    # plot attention weights for the first example of the batch; USE ONLY FOR DEV where len(predicted_y)=len(gold_y)
    # X is a tuple, first element is a tensor of size [batch_size, x_seq_len] 
    # y is a tuple, first element is a tensor of size [batch_size, y_seq_len]
    # attention_weights is a tensor of [batch_size, y_seq_len, x_seq_len] elements
    
    # X src labels for the first example
    input_labels = []
    X_list = X[0].cpu().tolist()    
    for id in X_list:
        input_labels.append(src_lookup.convert_ids_to_tokens(int(id)))
    
    # y tgt labels for the first example    
    y_list = y[0].cpu().tolist()
    output_labels = []
    for id in y_list:
        output_labels.append(tgt_lookup.convert_ids_to_tokens(int(id)))    
    
    # map weights
    data = attention_weights[0,:,:].t().cpu().tolist()
    #data = np.zeros((len(X_list), len(y_list)))
    #for i in range(len(data)): # each timestep i
    #    attention_over_inputs_at_timestep_i = attention_weights[i][0]
    #    data[:,i] = attention_over_inputs_at_timestep_i
    log_object.plot_heatmap(data, input_labels=input_labels, output_labels=output_labels, epoch=epoch, file_prefix="attention")

def _print_examples(log_object, model, loader, seq_len, src_lookup, tgt_lookup, skip_bos_eos_tokens = True):
    batch = iter(loader).next()
    if model.cuda:
        cuda_batch = []
        for batch_part in batch:                   
            cuda_batch_part = []
            for tensor in batch_part:
                cuda_batch_part.append(tensor.cuda())                    
            cuda_batch.append(tuple(cuda_batch_part))
        batch = tuple(cuda_batch)               
    
    X_sample, X_sample_lenghts, X_sample_mask, slots = batch[0][0], batch[0][1], batch[0][2], batch[0][3]
    y_sample, y_sample_lenghts, y_sample_mask = batch[1][0], batch[1][1], batch[1][2]
    seq_len = min(seq_len,len(X_sample))
    log_object.text("Printing {} examples (batch_size={}):".format(seq_len, len(X_sample)))
    
    X_sample = X_sample[0:seq_len]
    X_sample_lenghts = X_sample_lenghts[0:seq_len]
    X_sample_mask = X_sample_mask[0:seq_len]
     
    y_sample = y_sample[0:seq_len]
    y_sample_lenghts = y_sample_lenghts[0:seq_len]
    y_sample_mask = y_sample_mask[0:seq_len]
               
    model.eval()
    with torch.no_grad():
        output_tuple = model.run_batch(batch[0]) #batch[1]
        y_pred_sample = torch.argmax(output_tuple[0], dim=-1) 
        slot_accuracy = output_tuple[3]["accuracy_slt"]
        
    # print examples    
    for i in range(seq_len):  
        #print()
        """    
        #print("X   :", end='')
        lst = []
        for j in range(len(X_sample[i])):            
            lst.append(int(X_sample[i][j].item()))
        if skip_bos_eos_tokens:
            lst = clean_sequences([lst], src_lookup)[0]
        str = src_lookup.decode(lst, skip_bos_eos_tokens).replace(src_lookup.pad_token,"")
        log_object.text("X   :"+str, display = True)
        """
        print("Y   :", end='')
        lst = []
        for j in range(len(y_sample[i])):
            #token = int(y_sample[i][j].item())
            lst.append(int(y_sample[i][j].item()))
            #print(tgt_lookup.convert_ids_to_tokens(token) + " ", end='')        
        if skip_bos_eos_tokens:
            lst = clean_sequences([lst], tgt_lookup)[0]
        str = tgt_lookup.decode(lst, skip_bos_eos_tokens)
        print("\033[92m"+str+"\033[0m")
        log_object.text("Y   :"+str, display = False)
        
        print("PRED:", end='')
        lst = []
        for j in range(len(y_pred_sample[i])):
            #token = int(y_pred_sample[i][j].item())
            lst.append(int(y_pred_sample[i][j].item()))
            #print(tgt_lookup.convert_ids_to_tokens(token) + " ", end='')
        if skip_bos_eos_tokens:
            lst = clean_sequences([lst], tgt_lookup)[0]
        str = tgt_lookup.decode(lst, skip_bos_eos_tokens)
        print("\033[93m"+str+"\033[0m")
        log_object.text("PRED:"+str, display = False)        
        print("-" * 40)        
    log_object.text("Estimated examples batch slot accuracy: {}".format(slot_accuracy), display = True)
    

def train(model, train_loader, valid_loader=None, test_loader=None, model_store_path=None,
          resume=False, max_epochs=100000, patience=10, optimizer=None, criterion=None, lr_scheduler=None,
          tf_start_ratio=0., tf_end_ratio=0., tf_epochs_decay=0): # teacher forcing parameters
    if model_store_path is None: # saves model in the same folder as this script
        model_store_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)    
    
    log_path = os.path.join(model_store_path,"log")
    log_object = Log(log_path, clear=True)
    log_object.text("Training model: "+model.__class__.__name__)
    log_object.text("\tresume={}, patience={}, teacher_forcing={}->{} in {} epochs".format(resume, patience, tf_start_ratio, tf_end_ratio, tf_epochs_decay), display = False)
    total_params = sum(p.numel() for p in model.parameters())/1000
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000
    log_object.text("\ttotal_parameters={:.1f}M, trainable_parameters={:.1f}M".format(total_params, trainable_params))
    log_object.text(model.__dict__, display = False)
    log_object.text("", display = False)
        
    print("Working in folder [{}]".format(model_store_path))
    
    if not criterion:   
        criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_lookup.convert_tokens_to_ids(model.tgt_lookup.pad_token))
    
    n_class = len(model.tgt_lookup)
    batch_size = len(train_loader.dataset.X[0])
    current_epoch = 0
    current_patience = patience
    current_epoch_time = "?"
    current_epoch_train_time = "?"
    current_epoch_dev_time = "?"
    current_epoch_test_time = "?"
    best_accuracy = 0.

    # Calculates the decay per epoch. Returns a vector of decays.
    if tf_epochs_decay > 0:
        epoch_decay = np.linspace(tf_start_ratio, tf_end_ratio, tf_epochs_decay)

    if resume: # load checkpoint         
        extra_variables = model.load_checkpoint(model_store_path, extension="last")                
        load_optimizer_checkpoint(optimizer, model.cuda, model_store_path, extension="last")
        if "epoch" in extra_variables:
            current_epoch = extra_variables["epoch"]                        
        log_object.text("Resuming training from epoch {}".format(current_epoch))        
    
    while current_patience > 0 and current_epoch < max_epochs:        
        #mem_report()
        print("_"*120+"\n")             
        
        # teacher forcing ratio for current epoch
        tf_ratio = tf_start_ratio
        if tf_epochs_decay > 0:
            if current_epoch < tf_epochs_decay: 
                tf_ratio = epoch_decay[current_epoch]
            else: 
                tf_ratio = tf_end_ratio        
       
        
        log_object.text("")
        log_object.text("Starting epoch {}: current_patience={}, time_per_epoch={} ({}/{}/{}), tf_ratio={:.4f} ".format(current_epoch, current_patience,  current_epoch_time, current_epoch_train_time, current_epoch_dev_time, current_epoch_test_time, tf_ratio) )
        
        # train
        time_start = time.time()
        model.train()
        model.start_epoch(current_epoch)
        
        total_loss, gen_loss, slot_loss, att_loss, slot_acc = 0, 0, 0, 0, 0
        t = tqdm(train_loader, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [train]", unit="b") #ncols=120,
        for batch_index, batch in enumerate(t):                    
            if model.cuda:
                cuda_batch = []
                for batch_part in batch:                   
                    cuda_batch_part = []
                    for tensor in batch_part:
                        cuda_batch_part.append(tensor.cuda())                    
                    cuda_batch.append(tuple(cuda_batch_part))
                batch = tuple(cuda_batch)                
            # batch ususally is ((x_batch, x_batch_lenghts, x_batch_mask), (y_batch, y_batch_lenghts, y_batch_mask))
                        
            optimizer.zero_grad()
            
            output, loss, attention_weights, display_variables, decoder_dict = model.run_batch(batch[0], batch[1], criterion, tf_ratio)
            
            #a = list(model.text2slot.parameters())[0].clone()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)                        
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
             
            total_loss += loss.item()
            
            #b = list(model.text2slot.parameters())[0].clone()
            #print(torch.equal(a.data, b.data))
            
            if "lgen" in display_variables:
                gen_loss += display_variables["lgen"]
            if "latt" in display_variables:
                att_loss += display_variables["latt"]    
            if "lslt" in display_variables:
                slot_loss += display_variables["lslt"]
            if "accuracy_slt" in display_variables:
                slot_acc += display_variables["accuracy_slt"]
                
            current_scheduler_lr = "-"
            if lr_scheduler is not None:
                current_scheduler_lr = lr_scheduler.get_lr()[0]
            
            # update progress bar
            t_display_dict = OrderedDict()
            #if isinstance(display_variables, dict):
            #    for key in display_variables:
            #        t_display_dict[key] = display_variables[key]                                 
            #t_display_dict["cur_loss"] = loss.item()
            t_display_dict["gen"] = gen_loss / (batch_index+1)   
            t_display_dict["att"] = att_loss / (batch_index+1)   
            t_display_dict["slt"] = slot_loss / (batch_index+1) 
            t_display_dict["accslt"] = slot_acc / (batch_index+1)             
            t_display_dict["loss"] = total_loss / (batch_index+1)  
            t_display_dict["x/y"] = str(len(batch[0][0][0]))+"/"+str(len(batch[1][0][0]))
            t.set_postfix(ordered_dict = t_display_dict)
           
            #del output, batch, loss 
            #if batch_index%100==0:
            #    print(decoder_dict['p_gen'][0])
            #break
        
        #_plot_pgen_activations(batch[0], batch[1], model.src_lookup, model.tgt_lookup, decoder_dict, current_epoch, log_object)
            
        model.end_epoch()
        t.close()
        del t
        gc.collect()        
        
        if model.cuda:
            torch.cuda.empty_cache()
        
        log_object.text("\ttraining_loss={}".format(total_loss / len(train_loader)), display = False)
        log_object.var("Loss|Train loss|Validation loss", current_epoch, total_loss / len(train_loader), y_index=0)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, total_loss / len(train_loader), y_index=0)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, gen_loss / len(train_loader), y_index=1)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, att_loss / len(train_loader), y_index=2)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, slot_loss / len(train_loader), y_index=3)
        time_train = time.time() - time_start

        # dev        
        time_start = time.time()
        
        model.eval()
        with torch.no_grad():
            _print_examples(log_object, model, valid_loader, batch_size, model.src_lookup, model.tgt_lookup)
            
            total_loss, gen_loss, slot_loss, att_loss, slot_acc = 0, 0, 0, 0, 0
            t = tqdm(valid_loader, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="b")
            y_gold = list()
            y_predicted = list()
            
            for batch_index, batch in enumerate(t):
                if model.cuda:
                    cuda_batch = []
                    for batch_part in batch:                   
                        cuda_batch_part = []
                        for tensor in batch_part:
                            cuda_batch_part.append(tensor.cuda())                    
                        cuda_batch.append(tuple(cuda_batch_part))
                    batch = tuple(cuda_batch)               
                
                x_batch = batch[0][0]
                x_batch_lenghts = batch[0][1]
                x_batch_mask = batch[0][2]
                y_batch = batch[1][0]
                y_batch_lenghts = batch[1][1]
                y_batch_mask = batch[1][2]
            
                output, loss, batch_attention_weights, display_variables, decoder_dict = model.run_batch(batch[0], batch[1], criterion, tf_ratio=0.)
        
                y_predicted_batch = output.argmax(dim=2)
                y_gold += y_batch.tolist()
                y_predicted += y_predicted_batch.tolist()                
                
                total_loss += loss.data.item()
                
                # update progress bar
                if "lgen" in display_variables:
                    gen_loss += display_variables["lgen"]
                if "latt" in display_variables:
                    att_loss += display_variables["latt"]    
                if "lslt" in display_variables:
                    slot_loss += display_variables["lslt"]
                if "accuracy_slt" in display_variables:
                    slot_acc += display_variables["accuracy_slt"]
                    
                t_display_dict = OrderedDict()
                t_display_dict["gen"] = gen_loss / (batch_index+1)   
                t_display_dict["att"] = att_loss / (batch_index+1)   
                t_display_dict["slt"] = slot_loss / (batch_index+1)
                t_display_dict["accslt"] = slot_acc / (batch_index+1)                     
                t_display_dict["loss"] = total_loss / (batch_index+1)      
        
                #if isinstance(display_variables, dict):
                #    for key in display_variables:
                #        t_display_dict[key] = display_variables[key]                     
                t.set_postfix(ordered_dict = t_display_dict)
                
            if model.cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        log_object.text("\tvalidation_loss={}".format(total_loss / len(train_loader)), display = False)
        log_object.var("Loss|Train loss|Validation loss", current_epoch, total_loss / len(train_loader), y_index=1)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, total_loss / len(train_loader), y_index=4)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, gen_loss / len(valid_loader), y_index=5)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, att_loss / len(valid_loader), y_index=6)
        log_object.var("Detailed losses|T-loss|T-gen_loss|T-att_loss|T-slt_loss|V-loss|V-gen_loss|V-att_loss|V-slt_loss", current_epoch, slot_loss / len(valid_loader), y_index=7)
        
        total_score = 3 * slot_acc / len(valid_loader) 
        if True: #current_epoch%5==0:
            score, eval = evaluate(y_gold, y_predicted, model.tgt_lookup, cut_at_eos=True, use_accuracy=False, use_bleu=True)            
            total_score = (total_score + score)/4
            
            log_object.var("Average Scores|Dev scores|Test scores", current_epoch, score, y_index=0)            
            log_object.var("Sequence Accuracy Scores|Dev scores|Test scores", current_epoch, eval["sar"], y_index=0)            
            
            log_object.text("\tValidation scores: BLEU={:.3f} , METEOR={:.3f} , ROUGE-L(F)={:.3f}, SAR={:.3f}, average={:.3f}".format(eval["bleu"], eval["meteor"], eval["rouge_l_f"], eval["sar"], score))
            log_object.text("\tValidation estimated accuracy score = {:.3f}".format( slot_acc / len(valid_loader) ), display = False)
            
       
        if total_score > best_accuracy: 
            log_object.text("\tBest score = {:.4f}".format(total_score))
            best_accuracy = total_score
            model.save_checkpoint(model_store_path, extension="best", extra={"epoch":current_epoch})
            save_optimizer_checkpoint (optimizer, model_store_path, extension="best")            
            current_patience = patience
        
        # batch_attention_weights is a list of [batch_size, dec_seq_len, enc_seq_len] elements, where dim=2 is the softmax distribution for that decoder timestep            
        _plot_attention_weights(x_batch, y_batch, model.src_lookup, model.tgt_lookup, batch_attention_weights, current_epoch, log_object)
        
        # plot copy activations
        #_plot_copy_activations(x_batch, y_batch, model.src_lookup, model.tgt_lookup, decoder_dict, current_epoch, log_object)
        
        # dev cleanup
        t.close()
        del t, y_predicted_batch, y_gold, y_predicted       
        time_dev = time.time() - time_start
        # end dev
        
        # start test 
        time_test = 0
        # end test
        
        # end of epoch
        log_object.draw()
        #log_object.draw(last_quarter=True) # draw a second graph with last 25% of results
        
        model.save_checkpoint(model_store_path, "last", extra={"epoch":current_epoch})        
        save_optimizer_checkpoint (optimizer, model_store_path, extension="last")
       
        current_epoch += 1
        current_patience -= 1
        current_epoch_time = pretty_time(time_train+time_dev+time_test)
        current_epoch_train_time = pretty_time(time_train)
        current_epoch_dev_time = pretty_time(time_dev)
        current_epoch_test_time = pretty_time(time_test)

def save_optimizer_checkpoint (optimizer, folder, extension):
    filename = os.path.join(folder, "checkpoint_optimizer."+extension)
    #print("Saving optimizer parameters to {} ...".format(filename))    
    torch.save(optimizer.state_dict(), filename)    


def load_optimizer_checkpoint (optimizer, cuda, folder, extension):
    filename = os.path.join(folder, "checkpoint_optimizer."+extension)
    if not os.path.exists(filename):
        print("\tOptimizer parameters not found, skipping initialization")
        return
    print("Loading optimizer parameters from {} ...".format(filename))    
    optimizer.load_state_dict(torch.load(filename))
    if cuda:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
