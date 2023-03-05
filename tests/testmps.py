import torch
import torch.nn as nn

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())

# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

dev="mps"

class BiLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.lstm = nn.LSTM(
            input_size = 32,
            hidden_size = 64,
            bidirectional=True
        )

    def forward(self, X, xs_len):
        E = torch.transpose(self.embed(X), 0, 1)
        # E = torch.rand(15, 8, 32).to(dev)

        # packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(E, xs_len, enforce_sorted=False)
        # out, _ = self.lstm(packed_padded_x)
        # out, os_len = torch.nn.utils.rnn.pad_packed_sequence(out)

        out, _ = lstm(E)
        # repr = torch.transpose(out, 0, 1).max(dim=1)[0]
        # os_len = torch.randint(low=1, high=15, size=(8, ))
        out = out.reshape(out.size(0), 8, 2, -1)

        last_forward = out[-1, :, 0, :]
        first_backward = out[0, :, 1, :]
        repr = torch.concat([last_forward, first_backward], dim=-1)
        return repr

embed = nn.Embedding(100, 32).to(dev)
lstm = nn.LSTM(
    input_size = 32,
    hidden_size = 64,
    bidirectional=True
).to(dev)
criterion = nn.MSELoss()

X = torch.randint(high=100, size=(8, 15)).to(dev)
xs_len = torch.randint(low=1, high=15, size=(8, ))
y = torch.rand(8, 64*2).to(dev)

model=BiLSTM().to(dev)

repr = model(X, xs_len)

# E = torch.transpose(embed(X), 0, 1)
# E = torch.rand(15, 8, 32).to(dev)

# packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(E, xs_len, enforce_sorted=False)
# out, _ = lstm(packed_padded_x)
# out, os_len = torch.nn.utils.rnn.pad_packed_sequence(out)

# out, _ = lstm(E)
# repr = torch.transpose(out, 0, 1).max(dim=1)[0]
# out = out.reshape(max(os_len), 8, 2, -1)

# last_forward = torch.stack([o[o_len-1, 0, :] for o, o_len in zip(torch.transpose(out, 0, 1), os_len)])
# first_backward = out[0, :, 1, :]
# repr = torch.concat([last_forward, first_backward], dim=-1)


loss = criterion(repr, y)
print(loss)
loss.backward()
