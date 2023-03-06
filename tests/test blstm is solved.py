import torch
import torch.nn as nn

print("Torch version: ", torch.__version__)
torch.manual_seed(123)

L = 3
E = 2
H = 2
B = 3

embeddings = torch.rand(L, B, E)

model = nn.LSTM(input_size=E, hidden_size=H, bidirectional=True)
print(model(embeddings)[0])

model.to("mps")
print(model(embeddings.to("mps"))[0].cpu())
