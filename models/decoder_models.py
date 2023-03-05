import torch
import torch.nn as nn



class DecoderLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            batch_first = False
        )
        self.out = nn.Linear(opt['hidden_size'], opt['output_size'])
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, embeddings, state):

        output, new_state = self.lstm(torch.transpose(embeddings, 0, 1), state)
        output = self.softmax(self.out(torch.transpose(output, 0, 1)))

        return output, new_state


DECODERS = {
    'lstm': DecoderLSTM
}
DECODER_TYPES = list(DECODERS.keys())


if __name__ == '__main__':
    from torch import optim
    import random
    import copy

    I = 5
    L = 4
    E = 2
    H = 3
    B = 2

    def grad_norms(model):
        sum_grads = 0
        for p in model.parameters():
            # print("\t{:<30} {}".format(str(p.grad.shape), p.grad.norm()))
            sum_grads += p.grad.norm()
        return sum_grads

    def print_params(model):
        for p in model.parameters():
            print("{:<8} {}".format(str(p.shape), p.data))

    def print_results(results):
        line = '-' * 31
        print(line)
        print("{:<4} {:<4} {:>10} {:>10}".format('dev', 'dec', 'loss', 'grads'))
        print(line)
        for r in results:
            print("{:<4} {:<4} {:10.4f} {:10.4f}".format(r['dev'], r['dec'], r['loss'], r['grads']))
        print(line)

    embedding = torch.arange(start=0.0, end=1.0, step=1/I).unsqueeze(dim=1).expand(-1, E)

    decoder_type = 'lstm'
    decoder_opts = {
        "input_size": I,
        "embedding_size": E,
        "hidden_size": H,
        "output_size": I
    }

    random.seed(42)
    torch.manual_seed(42)

    criterion = nn.NLLLoss()

    embeds = torch.rand(B, 1, E)
    hidden = torch.rand(1, B, H)
    state = torch.rand(1, B, H)
    y = torch.randint(high=I, size=(B, 1))
    labels = torch.randint(high=I, size=(B, L))

    results = []

    basemodel = DECODERS[decoder_type](decoder_opts)
    for dev in ['cpu', 'mps']:

        model = copy.deepcopy(basemodel).to(dev)
        # print_params(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        ed = embeds.to(dev)
        hd = hidden.to(dev)
        sd = state.to(dev)
        yd = y.to(dev)

        optimizer.zero_grad()
        out, (new_h, new_s) = model(ed, (hd, sd))
        print("Out: ", out.cpu())

        loss = criterion(torch.transpose(out, 1, 2), yd)
        loss.backward()

        all_grads = grad_norms(model)

        results.append({'dev': dev, 'dec': decoder_type, 'loss': loss, 'grads': all_grads})

    print("Single step")
    print_results(results)


    results = []
    for dev in ['cpu', 'mps']:

        model = copy.deepcopy(basemodel).to(dev)
        hd = hidden.to(dev)
        ld = labels.to(dev)
        embedding = embedding.to(dev)
        optimizer.zero_grad()
        decoder_input = torch.full(size=(B, 1), fill_value=0, device=dev)
        state = (hd, torch.zeros((1, B, H), device=dev))
        teacher_forcing = False

        output = []
        for i in range(L):
            embeds = embedding[decoder_input].to(dev)
            out, state = model(embeds, state)
            output.append(out)
            if teacher_forcing:
                decoder_input = ld[:, i].view(B, 1)
            else:
                decoder_input = out.argmax(dim=-1)

        out = torch.stack(output, dim=-1).reshape(B, len(output), -1)
        print("Out: ", out.cpu())

        loss = criterion(torch.transpose(out, 1, 2), ld)
        loss.backward()

        all_grads = grad_norms(model)

        results.append({'dev': dev, 'dec': decoder_type, 'loss': loss, 'grads': all_grads})
        
    print("Multiple steps: ", L)
    print_results(results)
