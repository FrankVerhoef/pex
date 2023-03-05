import torch

print("Torch version: ", torch.__version__)
m1 = torch.rand(3,4,2)
m2 = torch.rand(3,5,2)
for m in (m1, m2):
    print('='*40)
    print(m)
    for device in ['cpu', 'mps']:
        print('-' * 40)
        print("Mean of slice on ", device)
        print('-' * 40)
        m = m.to(device)
        for i in range(3):
            print("Slice ", i)
            print(m[i][:2])
            print("Mean with dim=0: ", m[i][:2].mean(dim=0))
            print("Mean with dim=1: ", m[i][:2].mean(dim=1))