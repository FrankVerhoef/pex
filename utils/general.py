###
### General util functions
###

def savename(args):
    name = args.save
    if args.model == "bert":
        name += "bert"
    elif args.model[-4:] == "bart":
        name += args.model
    elif args.model == "seq2seq":
        name += "_seq2seq_{}_{}_I{}_E{}_H{}".format(
            args.encoder,
            args.decoder,
            args.vocab_size,
            args.embedding_size,
            args.hidden_size
        )
    return name

def print_params(model):
    print("Model parameters")
    for p in model.parameters():
        print("\t{:<30} {}".format(str(p.data.shape), p.data.norm()))

def print_grads(model):
    print("Gradients")
    for p in model.parameters():
        print("\t{:<30} {}".format(str(p.grad.shape), p.grad.norm()))

