import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class Encoder(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()

        self.hidden_size = 768

        #set Bert configuration to output attention weights
        configuration = BertConfig()
        configuration.output_attentions = True

        self.bertmodel = BertModel(configuration)
        self.bertmodel.resize_token_embeddings(vocab_size)
        for param in self.bertmodel.parameters():
            param.requires_grad = False

        self.device = device
        self.to(device)

    def forward(self, input_tuple):
        """
        Args:
            input_tuple (tensor): The input of the encoder. On the first position it must be a 2-D tensor of integers, padded. The second is the lenghts of the first.
                Shape: ([batch_size, seq_len], [batch_size], [att_mask]])

        Returns:
            Dict {
            'output': [batch_size, seq_len_enc, 768]
            'past': ((2, batch_size, num_heads, sequence_length, embed_size_per_head),(2, batch_size, num_heads, sequence_length, embed_size_per_head))
            'att' Att shape: ((batch_size, num_heads, sequence_length, sequence_length), (batch_size, num_heads, sequence_length, sequence_length))


        """
        self.bertmodel.eval()
        X, X_lengths, X_att_mask = input_tuple[0], input_tuple[1], input_tuple[2]
        batch_size = X.size(0)
        seq_len = X.size(1)

        output = torch.zeros(batch_size, seq_len, self.hidden_size).to(self.device)
        output.requires_grad = False

        with torch.no_grad():
            hidden_states, past, att = self.bertmodel(X, attention_mask=X_att_mask)
            for i in range(batch_size):
                output[i:i + 1, 0:X_lengths[i], :] = hidden_states[i:i + 1, 0:X_lengths[i], :]

        return {'output': output, 'past': past, 'att': att}