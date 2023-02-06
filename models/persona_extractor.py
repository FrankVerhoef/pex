import torch
import torch.nn as nn

from models.encoder_models import ENCODERS
from models.decoder_models import DECODERS


class PersonaExtractor(nn.Module):

    def __init__(self, encoder_type, encoder_opts, decoder_type, decoder_opts, start_token):
        super().__init__()
        self.encoder = ENCODERS[encoder_type](encoder_opts)
        self.decoder = DECODERS[decoder_type](decoder_opts)
        self.start_token = start_token


    def forward(self, xs, xs_len, max=20, teacher_forcing=False, ys=None):

        # Check validity of input
        if teacher_forcing:
            assert ys is not None, "Need to supply target sequence for teacher forcing"
            max = ys.size(1)
        
        # Apply encoder to input sequences
        encoded = self.encoder(xs, xs_len)

        # Set initial state for decoder
        B, H = encoded.shape
        decoder_input = torch.full(size=(B, 1), fill_value=self.start_token, device=encoded.device)
        state = (encoded.unsqueeze(dim=0), torch.zeros((1, B, H), device=encoded.device))

        # Generate tokens based on output of encoder
        output = []
        for i in range(max):
            out, state = self.decoder(decoder_input, state)
            output.append(out)
            if teacher_forcing:
                decoder_input = ys[:, i].view(B, 1)
            else:
                decoder_input = out.argmax(dim=-1)

        return torch.stack(output, dim=-1).reshape(B, len(output), -1)
  

if __name__ == '__main__':

    I = 60
    L = 20
    E = 16
    H = 48
    B = 8

    dev="mps"

    encoder_type = 'poolbilstm'
    encoder_opts = {
        "input_size": I,
        "embedding_size": E,
        "hidden_size": H,
        "aggregate_method": "max"
    }
    decoder_type = 'lstm'
    decoder_opts = {
        "input_size": I,
        "embedding_size": E,
        "hidden_size": {
            "mean": E,
            "lstm": H,
            "bilstm": H * 2,
            "poolbilstm": H * 2            
        }[encoder_type],
        "output_size": I
    }
    model = PersonaExtractor(encoder_type, encoder_opts, decoder_type, decoder_opts, start_token=1).to(dev)
    criterion = nn.NLLLoss()

    X = torch.randint(high=I, size=(B, L)).to(dev)
    X_lens = torch.randint(low=1, high=L, size=(B, ))
    y = torch.randint(high=I, size=(B, L)).to(dev)

    out = model(X, X_lens, teacher_forcing=True, ys=y)
    print("Shapes out: {}, y: {}".format(out.shape, y.shape))
    loss = criterion(out.transpose(1,2), y)
    print("Loss: ", loss)
    loss.backward()

