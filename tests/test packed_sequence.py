import torch
import torch.nn as nn

print("Torch version: ", torch.__version__)
torch.manual_seed(123)

L = 3
E = 2
H = 2
B = 1
seq_lengths = [L] * B  # For simplicity all equal.

embeddings = torch.rand(L, B, E)
packed_padded= torch.nn.utils.rnn.pack_padded_sequence(embeddings, seq_lengths)
model = nn.LSTM(input_size=E, hidden_size=H, bidirectional=True)

print("Output of packed embeddings on CPU")
print(model(packed_padded)[0])

model.to("mps")
embeddings = embeddings.to("mps")
packed_padded = packed_padded.to("mps")

print("Output of regular embeddings on MPS")
print(model(embeddings)[0].cpu())

print("Output of packed embeddings on MPS")
print(model(packed_padded)[0].cpu())
