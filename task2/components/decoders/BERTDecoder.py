import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        #init model to behave as decoder
        configuration = BertConfig()
        configuration.is_decoder = True
        configuration.output_attentions = True

        self.bertmodel = BertModel(configuration)
        self.bertmodel.resize_token_embeddings(vocab_size)  # resize the size of vocab to include new tokens
        for param in self.bertmodel.parameters():
            param.requires_grad = False

        self.lin_out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.device = device
        self.to(device)

    def forward(self, y_tuple, X_att_mask, encoder_hidden_states):
        """
        Args:
            y_tuple (tensor): The input of the decoder. On the first position it must be a 2-D tensor of integers, padded. The second is the lenghts of the first.
                Shape: ([batch_size, seq_len], [batch_size], [att_mask]])
            encoder_hidden_states (tensor): [batch_size, sequence_length, hidden_size] - HS at last layer of the encoder.


        Returns:
            Dict {
            'output' : [batch_size, y_seq_len_enc, vocab_size]
            'past' (tuple): ((2, batch_size, num_heads, sequence_length, embed_size_per_head),(2, batch_size, num_heads, sequence_length, embed_size_per_head))
            'att' Att shape (tuple): ((batch_size, num_heads, sequence_length, sequence_length), (batch_size, num_heads, sequence_length, sequence_length))


        """
        y = y_tuple[0].to(self.device)
        y_lenghts = y_tuple[1].to(self.device)
        y_att_mask = y_tuple[2].to(self.device)
        batch_size = y.size(0)
        y_seq_len = y.size(1)

        output = torch.zeros(batch_size, y_seq_len, self.hidden_size).to(self.device)
        output.requires_grad = False
        #TODO: Fix encoder attention mask.
        with torch.no_grad():
            hidden, past, decoder_attention = self.bertmodel(y, attention_mask=y_att_mask,
                                                             encoder_hidden_states=encoder_hidden_states)
            for i in range(batch_size):
                output[i:i + 1, 0:y_lenghts[i], :] = hidden[i:i + 1, 0:y_lenghts[i], :]

        out_lin = self.lin_out(output)
        output = self.softmax(out_lin)

        return {'output': output, 'past': past, 'att': decoder_attention}