"""
    Code to make a sentence representation, using four different methods:
    1. MeanEmbedding: calculate the mean of the token embeddings
    2. Use LSTM to encode the tokens and use the last hidden state as sentence representation
    3. Use BiLSTM to encode the tokens and use the concatenation of the hidden state of last token of the forward pass 
        and the hidden state of first token of the backward pass as sentence representation
    4. Use BiLSTM to encode the tokens; then concatenate the forward and backward hidden states of each token, 
        and use pooling (max of average) over all tokens to create sentence representation

    All modules are initialised using a dict 'opt' with the relevant parameters for the model

    The forward function in each model takes 'embeddings' and 'xs-len' as input. 
        embeddings: tensor with batch of embeddings, shape (B, L, E)
        seq_lengths: tensor with the length of each sequence in the batch, shape (B, )
    Output has shape (B, repr_size)
    """

import torch
import torch.nn as nn
import utils.logging as logging


class MeanEmbedding(nn.Module):

    def __init__(self, opt):
        super().__init__()

    def forward(self, embeddings, seq_lengths):

        # calculate the mean embedding of each sequence, excluding the padded positions
        repr = torch.stack([e[:e_len].mean(dim=-2) for e, e_len in zip(embeddings, seq_lengths)])
        return repr

class UniLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.embedding_size = opt['embedding_size']
        self.hidden_size = opt['hidden_size']
        self.lstm = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = False,
            bidirectional = False
        )

    def forward(self, embeddings, seq_lengths):

        # pack, run through LSTM --> pack_padded_sequece seems to give error or incorrect output on MPS
        if True: #embeddings.is_mps:
            output, _ = self.lstm(torch.transpose(embeddings, 0, 1))
        else:
            packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(embeddings, 0, 1), seq_lengths, batch_first=False, enforce_sorted=False)
            output, _ = self.lstm(packed_padded_x)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False)

        # shape of output is L, B, H; use last hidden state per sequence as representation
        repr = torch.stack([
            seq[seq_len-1, :] 
            for seq, seq_len in zip(torch.transpose(output, 0, 1), seq_lengths)
        ])

        return repr

class BiLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.embedding_size = opt['embedding_size']
        self.hidden_size = opt['hidden_size']
        self.lstm = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = False,
            bidirectional = True
        )

    def forward(self, embeddings, seq_lengths):

        # input shape is B, L, E
        B, L, E = embeddings.shape

        # pack, run through LSTM --> pack_padded_sequece seems to give error or incorrect output on MPS
        if True: #embeddings.is_mps:
            output, _ = self.lstm(torch.transpose(embeddings, 0, 1))
            max_L = L
        else:
            packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(embeddings, 0, 1), seq_lengths, batch_first=False, enforce_sorted=False)
            output, _ = self.lstm(packed_padded_x)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
            max_L = seq_lengths.max()

        logging.debug("BiLSTM out ".format(embeddings.device))
        logging.debug(output)
        # shape of output is max-L, B, 2 x H --> transpose B and L, and then reshape to separate hidden dimensions of forward and backward pass
        output = torch.transpose(output, 0, 1).reshape(B, max_L, 2, -1)

        # use concat of last forward and first backward hidden state as representation
        last_forward = torch.stack([seq[seq_len-1, 0, :] for seq, seq_len in zip(output, seq_lengths)])
        first_backward = output[:, 0, 1, :]
        repr = torch.cat((last_forward, first_backward), dim=-1)

        return repr


class PoolBiLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        assert opt["aggregate_method"] in ["max", "avg"], "Invalid aggregation method: {}".format(opt["aggregate_method"])
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = 1,
            batch_first = False,
            bidirectional = True
        )
        self.aggregate_method = opt['aggregate_method']

    def forward(self, embeddings, seq_lengths):
        
        # pack, run through LSTM --> pack_padded_sequece seems to give error on MPS with BiLSTM
        if True: #embeddings.is_mps:
            output, _ = self.lstm(torch.transpose(embeddings, 0, 1))
        else:
            packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(embeddings, 0, 1), seq_lengths, batch_first=False, enforce_sorted=False)
            output, _ = self.lstm(packed_padded_x)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False)

        # shape of output is L, B, 2 x H --> first transpose B and L
        output = torch.transpose(output, 0, 1)

        # perform pooling over the layers, make sure to only include values up to seq_lengths
        if self.aggregate_method == "max":
            if False: #output.is_mps:
                repr = torch.stack([
                    torch.stack([o for o in seq[:o_len]]).max(dim=0)[0]
                    for seq, o_len in zip(output, seq_lengths)
                ])
            else:
                repr = torch.stack([seq[:seq_len, :].max(dim=0)[0] for seq, seq_len in zip(output, seq_lengths)])
        elif self.aggregate_method == "avg":
            repr = torch.stack([seq[:seq_len, :].mean(dim=0) for seq, seq_len in zip(output, seq_lengths)])
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

if __name__ == '__main__':
    from torch import optim
    import random
    import copy

    logging.set_log_level(logging.INFO)

    L = 5
    E = 2
    H = 3
    B = 4

    def grad_norms(model):
        sum_grads = 0
        for p in model.parameters():
            sum_grads += p.grad.norm()
        return sum_grads

    def model_params(model):
        result = "MODEL PARAMS\n"
        for p in model.parameters():
            result += "{:<8} {}\n".format(str(p.shape), p.data.cpu())
        return result

    def format_results(results):

        line = '-' * 38 + '\n'
        result = "RESULTS\n"
        result += line
        result += "{:<4} {:<11} {:>10} {:>10}\n".format('dev', 'enc', 'loss', 'grads')
        result += line
        for r in results:
            result += "{:<4} {:<11} {:10.4f} {:10.4f}\n".format(r['dev'], r['enc'], r['loss'], r['grads'])
        result += line
        return result


    encoder_opts = {
        "embedding_size": E,
        "hidden_size": H,
        "aggregate_method": "max"
    }

    random.seed(42)
    torch.manual_seed(42)

    criterion = nn.MSELoss()

    embeddings = torch.rand(B, L, E)
    hidden = torch.zeros(1, B, H)
    state = torch.zeros(1, B, H)
    X_lens = torch.randint(low=1, high=L+1, size=(B, ))

    results = []

    for encoder_type in ENCODER_TYPES:

        basemodel = ENCODERS[encoder_type](encoder_opts)
        logging.spam(model_params(basemodel))
        output_size = {
            "mean": E,
            "lstm": H,
            "bilstm": H * 2,
            "poolbilstm": H * 2            
        }[encoder_type]
        y = torch.zeros(B, output_size)

        for dev in ['cpu', 'mps']:

            model = copy.deepcopy(basemodel).to(dev)
            logging.spam(model_params(model))
            if encoder_type != 'mean':
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            ed = embeddings.to(dev)
            yd = y.to(dev)

            if encoder_type != 'mean':
                optimizer.zero_grad()
            out = model(ed, X_lens)
            logging.debug("Out {}\n{}".format(encoder_type, out.cpu()))
            loss = criterion(out, yd)
            logging.debug("Loss: {}".format(loss))
            if encoder_type != 'mean':
                loss.backward()
                all_grads = grad_norms(model)
            else:
                all_grads = 0

            results.append({'dev': dev, 'enc': encoder_type, 'loss': loss, 'grads': all_grads})

    logging.report(format_results(results))