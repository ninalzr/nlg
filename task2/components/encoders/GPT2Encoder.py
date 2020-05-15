import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

class Encoder(nn.Module):
    def __init__(self, vocab_size, device):       
        super().__init__()
        
        self.hidden_size = 768
        self.vocab_size = vocab_size
        #self.hidden_size = vocab_size
        self.gpt2model = GPT2Model.from_pretrained('distilgpt2')
        self.gpt2model.resize_token_embeddings(self.vocab_size)
        for param in self.gpt2model.parameters():
            param.requires_grad = False
        self.output_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.device = device
        self.to(device)

    def forward(self, input_tuple):
        """
        Args:
            input_tuple (tensor): The input of the encoder. On the first position it must be a 2-D tensor of integers, padded. The second is the lenghts of the first.
                Shape: ([batch_size, seq_len_enc], [batch_size], other)
        Returns:
            A tuple containing the output and the states of the last LSTM layer. The states of the LSTM layer is also a
            tuple that contains the hidden and the cell state, respectively . 
                Output shape: [batch_size, seq_len_enc, 768]
        
        """
        self.gpt2model.eval()
        X, X_lengths = input_tuple[0], input_tuple[1]

        print(X_lengths)
        batch_size = X.size(0)
        seq_len = X.size(1)
        print(seq_len)
        output = torch.zeros(batch_size, seq_len, self.hidden_size).to(self.device)
        print("Encoder output shape {}".format(output.shape))
        output.requires_grad = False
        with torch.no_grad(): # hack ?? documentation is not clear on padding, so, skipping it with this hack
            hidden_states, past = self.gpt2model(X)
            print("HS size {}".format(hidden_states.shape))
            for i in range(batch_size):
                output[i:i+1, 0:X_lengths[i], :] = hidden_states[i:i+1, 0:X_lengths[i], :]

        lin_output = self.output_linear(output)
        bos_tensor = torch.zeros(batch_size, 1, self.vocab_size).to(self.device)
        outputs = torch.cat((bos_tensor, lin_output), dim=1)
        print("Output BOS out {}".format(outputs.shape))
        return outputs #{'output':output}