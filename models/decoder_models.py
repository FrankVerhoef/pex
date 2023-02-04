import torch
import torch.nn as nn



class LSTM(nn.Module):

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
        self.out = nn.Linear(opt['hidden_size'], opt['output_size'])
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, xs, state):

        embeds = self.embed(xs)
        output, new_state = self.lstm(embeds, state)
        output = self.softmax(self.out(output))

        return output, new_state


DECODERS = {
    'lstm': LSTM
}
DECODER_TYPES = list(DECODERS.keys())
