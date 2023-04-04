###
### General util functions
###

import torch

def savename(args):
    name = args.save
    if args.model == "kg_gen":
        name += "_kgg"
    elif args.model == "bert":
        name += "_bert"
    elif args.model[-4:] == "bart":
        name += '_' + args.model
    elif args.model == "seq2seq":
        name += "_seq2seq_{}_{}_I{}_E{}_H{}".format(
            args.encoder,
            args.decoder,
            args.vocab_size,
            args.embedding_size,
            args.hidden_size
        )
    return name

def loadname_prefix(name):
    prefix_end = name.index('_')
    prefix = name[:prefix_end]
    return prefix

def print_params(model):
    print("Model parameters")
    for p in model.parameters():
        print("\t{:<30} {}".format(str(p.data.shape), p.data.norm()))

def print_grads(model):
    print("Gradients")
    for p in model.parameters():
        print("\t{:<30} {}".format(str(p.grad.shape), p.grad.norm()))

def padded_tensor(tensorlist, pad_value=0):
    return torch.nn.utils.rnn.pad_sequence(tensorlist, batch_first=True, padding_value=pad_value)

def padded_tensor_left(tensorlist, pad_value=0):
    max_len = max([len(t) for t in tensorlist])
    padded = torch.full((len(tensorlist), max_len), fill_value=pad_value)
    for i in range(len(tensorlist)):
        l = len(tensorlist[i])
        if l < max_len:
            padded[i, -l:] = tensorlist[i]
        else:
            padded[i, :] = tensorlist[i]
    return padded