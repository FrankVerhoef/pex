"""
    Code to make a sentence representation, using four different methods:
    1. MeanEmbedding: calculate the mean of the token embeddings
    2. Use LSTM to encode the tokens and use the last hidden state as sentence representation
    3. Use BiLSTM to encode the tokens and use the concatenation of the hidden state of last token of the forward pass 
        and the hidden state of first token of the backward pass as sentence representation
    4. Use BiLSTM to encode the tokens; then concatenate the forward and backward hidden states of each token, 
        and use pooling (max of average) over all tokens to create sentence representation

    All modules are initialised using a dict 'opt' with the relevant parameters for the model
"""

import torch
import torch.nn as nn


class MeanEmbedding(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.embed = nn.Embedding(opt["input_size"], opt["embedding_size"])

    def forward(self, xs, xs_len):

        # calculate the mean of each x, excluding the padded positions
        repr = torch.stack([self.embed(x)[:x_len].mean(dim=-2) for x, x_len in zip(xs, xs_len)])

        return repr

class UniLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.embed = nn.Embedding(opt["input_size"], opt["embedding_size"])
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )

    def forward(self, xs, xs_len):
        embeds = self.embed(xs)

        # pack, run through LSTM, then unpack
        packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(embeds, xs_len, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_padded_x)
        output, os_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # shape of output is B, L, H; use last hidden state per sequence as representation
        repr = torch.stack([o[o_len-1, :] for o, o_len in zip(output, os_len)])

        return repr

class BiLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.embed = nn.Embedding(opt["input_size"], opt["embedding_size"])
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

    def forward(self, xs, xs_len):
        embeds = self.embed(xs)

        # input shape is B, L, E
        B, L, E = embeds.shape

        # pack, run through LSTM
        packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(embeds, xs_len, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_padded_x)
        output, os_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # shape of output is B, max-L, 2 x H
        output = output.reshape(B, max(os_len), 2, -1)

        # use concat of last forward and first backward hidden state as representation
        last_forward = torch.stack([o[o_len-1, 0, :] for o, o_len in zip(output, os_len)])
        first_backward = output[:, 0, 1, :]
        repr = torch.concat([last_forward, first_backward], dim=-1)

        return repr


class PoolBiLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        assert opt["aggregate_method"] in ["max", "avg"], "Invalid aggregation method: {}".format(opt["aggregate_method"])
        self.embed = nn.Embedding(opt["input_size"], opt["embedding_size"])
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.aggregate_method = opt['aggregate_method']

    def forward(self, xs, xs_len):
        embeds = self.embed(xs)
        
        # pack, run through LSTM
        packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(embeds, xs_len, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_padded_x)
        output, os_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # shape of output is B, max-L, 2 x H
        # perform pooling over the layers, make sure to only include values up to xs_len
        if self.aggregate_method == "max":
            repr = torch.stack([o[:o_len, :].max(dim=0)[0] for o, o_len in zip(output, os_len)])
        elif self.aggregate_method == "avg":
            repr = torch.stack([o[:o_len, :].mean(dim=0) for o, o_len in zip(output, os_len)])
        else:
            repr = None # should never occur because of check at initialization

        return repr

ENCODERS = {
    'mean': MeanEmbedding,
    'lstm': UniLSTM,
    'bilstm': BiLSTM,
    'poolbilstm': PoolBiLSTM
}
ENCODER_TYPES = list(ENCODERS.keys())