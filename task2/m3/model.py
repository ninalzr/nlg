import  os
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):

    def __init__(self, src_lookup, tgt_lookup, encoder, decoder, device):
        super().__init__()

        self.src_lookup = src_lookup
        self.tgt_lookup = tgt_lookup
        self.src_bos_token_id = src_lookup.convert_tokens_to_ids(src_lookup.bos_token)
        self.src_eos_token_id = src_lookup.convert_tokens_to_ids(src_lookup.eos_token)
        self.tgt_bos_token_id = src_lookup.convert_tokens_to_ids(tgt_lookup.bos_token)
        self.tgt_eos_token_id = src_lookup.convert_tokens_to_ids(tgt_lookup.eos_token)
        self.vocab_size = len(tgt_lookup)

        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.to(self.device)


    def forward(self, X_tuple, y_tuple, teacher_forcing_ratio=0.):
        #TODO: train with tf ratio
        x, x_lenghts, x_mask = X_tuple[0], X_tuple[1], X_tuple[2]
        batch_size = x.shape[0]

        encoder_dict = self.encoder.forward((x, x_lenghts, x_mask))
        enc_output = encoder_dict["output"]
        enc_past = encoder_dict["past"]
        enc_att = encoder_dict["att"]

        decoder_dict = self.decoder.forward(y_tuple, X_att_mask=x_mask, encoder_hidden_states=enc_output)

        output_decoder = decoder_dict['output']
        attention_decoder = decoder_dict['att']

        return output_decoder, attention_decoder


    def run_batch(self, X_tuple, y_tuple=None, criterion=None):
        y = y_tuple[0].to(self.device)
        output_decoder, attention_decoder = self.forward(X_tuple, y_tuple, teacher_forcing_ratio=0.)

        total_loss = 0

        if criterion is not None:
            loss = criterion(output_decoder.view(-1, self.vocab_size), y.contiguous().flatten())
            total_loss += loss

        return output_decoder, total_loss, attention_decoder


    def load_checkpoint(self, folder, extension):
        filename = os.path.join(folder, "checkpoint." + extension)
        print("Loading model {} ...".format(filename))
        if not os.path.exists(filename):
            print("\tModel file not found, not loading anything!")
            # return {}
            raise Exception("Error, model file not found! {} -> model {}".format(folder, extension))

        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        return checkpoint["extra"]


    def save_checkpoint(self, folder, extension, extra={}):
        filename = os.path.join(folder, "checkpoint." + extension)
        checkpoint = {}
        checkpoint["state_dict"] = self.state_dict()
        checkpoint["extra"] = extra
        torch.save(checkpoint, filename)
