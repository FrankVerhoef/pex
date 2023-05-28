###
### General util functions
###

import torch
import json
from ast import literal_eval

def savename(args):
    name = args.save
    if args.model == "dialogpt":
        name += "_dgpt"
    elif args.model == "kg_gen":
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

def prettydict(d, title=None):
    max_keylen = max([len(str(k)) for k in d.keys()])
    result = title + '\n' if title is not None else ""
    result += '\n'.join([f"{key.ljust(max_keylen)} : {value}" for key, value in d.items()])
    return result

def dict_with_key_prefix(d, prefix=None):
    if prefix is None:
        return d
    else:
        return {prefix + str(k): v for k, v in d.items()}
    
def save_config(savepath, args):
    with open(savepath, 'w') as f:
        f.write(f"# Configfile action={args.action}, model={args.model}, task={args.task}\n")
        for k, v in vars(args).items():
            if k not in ['action', 'model', 'task', 'configfile']:
                f.write(f"{k} = {v}\n")

def load_config(savepath):
    config = {}
    with open(savepath, 'r') as f:

        # Read first line with main args; format = "# Configfile action={args.action}, model={args.model}, task={args.task}\n"
        lines = f.readlines()
        assert len(lines) > 0, "Missing header in configfile"
        assert lines[0][:12] == '# Configfile', f"Invalid header for configfile: '{lines[0]}'"
        mainargs = lines[0][:-1].replace(',', '').split()[-3:]
        for arg in mainargs:
            assert arg.find('=') >= 0, f"Invalid format: '{arg}'"
            k, v = arg.replace(' ', '').split('=')
            config[k] = v
        
        # Read the remaining lines; format = "{k} = {v}\n"
        for arg in lines[1:]:
            k, v = arg[:-1].replace(' ', '').split('=')
            try:
                config[k] = literal_eval(v)
            except:
                config[k] = v

    return config

def save_dict(savepath, dict):
    with open(savepath, 'w') as f:
        f.write(json.dumps(dict, indent=4))