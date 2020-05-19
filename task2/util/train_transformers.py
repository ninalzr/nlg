import os, time, gc
import torch
import torch.nn as nn

#from task2.util.log import Log
#from task2.util.validation_metrics import evaluate
from tqdm import tqdm


def train(model, train_loader, valid_loader, tgt_lookup, optimizer, criterion, max_epochs,
          lr, model_store_path = None):
    current_epoch = 0
    while current_epoch < max_epochs:

        model.train()

        total_loss = 0
        start_time = time.time()

        t = tqdm(train_loader, mininterval=0.5, desc="Epoch " + str(current_epoch) + " [train]", unit="b")
        for idx, t_batch in enumerate(t):
            X_tuple = t_batch[0]
            y_tuple = t_batch[1]

            output_decoder, loss, attention_decoder = model.run_batch(X_tuple, y_tuple, criterion=criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            total_loss += loss.item()


        t.close()
        del t
        gc.collect()

        if model.cuda:
            torch.cuda.empty_cache()


        #dev
        time_start = time.time()
        with torch.no_grad():
            y_gold = list()
            y_predicted = list()
            v = tqdm(valid_loader, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="b")
            for i, v_batch in enumerate(v):
                X_tuple = v_batch[0]
                y_tuple = v_batch[1]
                output_decoder, loss, attention_decoder = model.run_batch(X_tuple, y_tuple, criterion = criterion)
                y_predicted_batch = output_decoder.argmax(dim=2)
                y_gold += y_tuple[0].tolist()
                y_predicted += y_predicted_batch.tolist()

            seq_len = X_tuple[0].size(0)

            for i in range(seq_len):
                lst = []
                for j in range(len(y_predicted[i])):
                    lst.append(y_predicted[i][j])
                print("Y Pred: ")
                tstr = tgt_lookup.decode(lst, skip_bos_eos_tokens = True)
                print(tstr)

                glst = []
                print("Y Gold: ")
                for g in range(len(y_gold[i])):
                    glst.append(y_gold[i][g])
                gstr = tgt_lookup.decode(glst, skip_bos_eos_tokens = True)
                print(gstr)
            v.close()
            del v, y_predicted_batch, y_gold, y_predicted
            time_dev = time.time() - time_start
        current_epoch += 1

        model.save_checkpoint(model_store_path, "last", extra={"epoch": current_epoch})
        save_optimizer_checkpoint(optimizer, model_store_path, extension="last")


def train_t5(model, train_loader, valid_loader, tgt_lookup, optimizer, criterion, max_epochs,
          lr, model_store_path = None):
    current_epoch = 0
    while current_epoch < max_epochs:
        model.train()
        total_loss = 0
        start_time = time.time()

        t = tqdm(train_loader, mininterval=0.5, desc="Epoch " + str(current_epoch) + " [train]", unit="b")
        for idx, t_batch in enumerate(t):
            X_tuple = t_batch[0]
            y_tuple = t_batch[1]

            output, loss = model.forward(X_tuple, y_tuple, criterion=criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            total_loss += loss.item()


        t.close()
        del t
        gc.collect()

        if model.cuda:
            torch.cuda.empty_cache()


        #dev
        time_start = time.time()
        with torch.no_grad():
            y_gold = list()
            y_predicted = list()
            v = tqdm(valid_loader, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="b")
            for i, v_batch in enumerate(v):
                X_tuple = v_batch[0]
                y_tuple = v_batch[1]
                output, loss = model.forward(X_tuple, y_tuple, criterion = criterion)
                y_predicted_batch = output.argmax(dim=2)
                y_gold += y_tuple[0].tolist()
                y_predicted += y_predicted_batch.tolist()

            seq_len = X_tuple[0].size(0)

            for i in range(seq_len):
                lst = []
                for j in range(len(y_predicted[i])):
                    lst.append(y_predicted[i][j])
                print("Y Pred: ")
                tstr = tgt_lookup.decode(lst, skip_bos_eos_tokens = True)
                print(tstr)

                glst = []
                print("Y Gold: ")
                for g in range(len(y_gold[i])):
                    glst.append(y_gold[i][g])
                gstr = tgt_lookup.decode(glst, skip_bos_eos_tokens = True)
                print(gstr)
            v.close()
            del v, y_predicted_batch, y_gold, y_predicted
            time_dev = time.time() - time_start
        current_epoch += 1

        model.save_checkpoint(model_store_path, "last", extra={"epoch": current_epoch})
        save_optimizer_checkpoint(optimizer, model_store_path, extension="last")


def save_optimizer_checkpoint(optimizer, folder, extension):
    filename = os.path.join(folder, "checkpoint_optimizer." + extension)
    # print("Saving optimizer parameters to {} ...".format(filename))
    torch.save(optimizer.state_dict(), filename)



def load_optimizer_checkpoint(optimizer, cuda, folder, extension):
    filename = os.path.join(folder, "checkpoint_optimizer." + extension)
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



