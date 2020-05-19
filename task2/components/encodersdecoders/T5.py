import os
import torch
import torch.nn as nn
from transformers import T5Model, T5Config, T5ForConditionalGeneration


class T5(nn.Module):
    def __init__(self, hidden_size, vocab_size, device):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        configuration = T5Config()
        self.T5Model = T5Model(configuration)
        self.T5Model.resize_token_embeddings(vocab_size)  # resize the size of vocab to include new tokens
        for param in self.T5Model.parameters():
            param.requires_grad = False

        self.lin_out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.device = device
        self.to(device)

    def forward(self, X_tuple, y_tuple, criterion=None):
        X = X_tuple[0].to(self.device)
        X_len = X_tuple[1]
        X_mask = X_tuple[2].to(self.device)
        y = y_tuple[0].to(self.device)
        y_lenghts = y_tuple[1].to(self.device)
        batch_size = X.size(0)
        y_seq_len = y.size(1)

        output = torch.zeros(batch_size, y_seq_len, self.hidden_size).to(self.device)
        output.requires_grad = False
        with torch.no_grad():
            hidden = self.T5Model(input_ids=X, attention_mask=X_mask, decoder_input_ids=y)
            last_hidden = hidden[0]
            for i in range(batch_size):
                output[i:i + 1, 0:y_lenghts[i], :] = last_hidden[i:i + 1, 0:y_lenghts[i], :]

        out_lin = self.lin_out(output)
        output = self.softmax(out_lin)

        total_loss = 0
        if criterion is not None:
            loss = criterion(output.view(-1, self.vocab_size), y.contiguous().flatten())
            total_loss = loss

        return output, total_loss

    def load_checkpoint(self, folder, extension):
        filename = os.path.join(folder, "checkpoint." + extension)
        print("Loading model {} ...".format(filename))
        if not os.path.exists(filename):
            print("\tModel file not found, not loading anything!")
            # return {}
            raise Exception("Error, model file not found! {} -> model {}".format(folder, extension))

        checkpoint = torch.load(filename, map_location=self.device)
        # self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        # self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.load_state_dict(checkpoint["state_dict"])

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        return checkpoint["extra"]

    def save_checkpoint(self, folder, extension, extra={}):
        filename = os.path.join(folder, "checkpoint." + extension)
        checkpoint = {}
        # checkpoint["encoder_state_dict"] = self.encoder.state_dict()
        # checkpoint["decoder_state_dict"] = self.decoder.state_dict()
        checkpoint["state_dict"] = self.state_dict()
        checkpoint["extra"] = extra
        torch.save(checkpoint, filename)