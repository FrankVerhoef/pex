"""
python run/main.py
to run with defaults
"""
import torch
import torch.nn as nn
from torch import optim
import random
import wandb
import copy

from transformers import AutoTokenizer
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import BertClassifier
from dataset.msc_summary import MSC_Turns, extra_tokens
from dataset.vocab import Vocab, PAD_TOKEN, START_TOKEN
from utils.general import savename, print_grads, print_params



def train(model, trainloader, validloader, optimizer, criterion, device="cpu", epochs=1, log_interval=1000):

    train_losses = []
    valid_losses = []
    min_loss = None
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(iter(trainloader)):

            loss = model.train_step(batch, optimizer, criterion, device)
            train_losses.append(loss)
    
            if (step + 1) % log_interval == 0:
                loss_avg = sum(train_losses[-log_interval:]) / log_interval
                # wandb.log({
                #     "train_loss": loss_avg
                # })
                print("Epoch {}, step {}: loss={}".format(epoch, step+1, loss_avg))
    
        model.eval()
        valid_stats = eval(model, validloader, criterion, device)
        valid_losses.append(valid_stats['loss'])
        print("Validation stats: ", valid_stats)
        if (min_loss is None) or (valid_stats['loss'] < min_loss):
                min_loss = valid_stats['loss']
                print("Best loss reduced to: ", min_loss)
                best_model = copy.deepcopy(model)

    return best_model, {"train_loss": train_losses, "valid_loss": valid_losses}


def eval(model, dataloader, criterion, device):

    losses = []
    model.to(device)
    model.eval()

    for batch in iter(dataloader):

        stats = model.valid_step(batch, criterion, device)
        losses.append(stats["loss"])

    stats = {
        "loss": sum(losses) / len(losses),
    }
    # wandb.run.summary.update(test_stats)

    return stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--log_interval", type=int, default=10, help="report interval")
    parser.add_argument("--load", type=str, default="", help="filename of model to load")
    parser.add_argument("--save", type=str, default="", help="filename to save the model")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--task", type=str, default="classify", choices=["generate", "classify"])
    
    # Encoder and decoder model
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert"], help="Encoder model")
    parser.add_argument("--encoder", type=str, default="mean", help="Encoder model")
    parser.add_argument("--embedding_size", type=int, default=100, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--aggregate_method", type=str, default="avg", choices=["avg", "max"], help="Aggregate method for Pool Bi-LSTM")
    parser.add_argument("--decoder", type=str, default="lstm", help="Decoder model")
    parser.add_argument("--decoder_max", type=int, default=20, help="Max number of tokens to generate with decoder")

    # Dataset
    parser.add_argument("--datadir", type=str, default="/Users/FrankVerhoef/Programming/PEX/data/", help="Datadir")
    parser.add_argument("--traindata", type=str, default="msc/msc_personasummary/session_1/train.txt", help="Dataset file for training")
    parser.add_argument("--validdata", type=str, default="msc/msc_personasummary/session_1/valid.txt", help="Dataset file for validation")
    parser.add_argument("--testdata", type=str, default="msc/msc_personasummary/session_1/test.txt", help="Dataset file for testing")
    parser.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")
    parser.add_argument("--train_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--test_samples", type=int, default=None, help="Max number of test samples")
    parser.add_argument("--persona_tokens", type=bool, default=False, help="Whether to insert special persona token before each dialogue turn")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.task == 'classify':

        print("Set up {} to {}".format(args.model, args.task))
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        traindata = MSC_Turn_Facts(args.datadir + args.traindata, tokenizer, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.train_samples)
        validdata = MSC_Turn_Facts(args.datadir + args.validdata, tokenizer, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.test_samples)
        testdata = MSC_Turn_Facts(args.datadir + args.testdata, tokenizer, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.test_samples)
        if args.persona_tokens:
            num_added_toks = tokenizer.add_tokens(extra_tokens)
        model = BertClassifier()
        model.bert.resize_token_embeddings(len(tokenizer))
        criterion = nn.NLLLoss(reduction='mean')

    elif args.task == 'generate':
        print("Set up {} to {}".format(args.model, args.task))
        vocab = Vocab()
        traindata = MSC_Turns(args.datadir + args.traindata, vocab.text2vec, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.train_samples)
        validdata = MSC_Turns(args.datadir + args.validdata, vocab.text2vec, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.test_samples)
        testdata = MSC_Turns(args.datadir + args.testdata, vocab.text2vec, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.test_samples)
        if args.persona_tokens:
            vocab.add_special_tokens(extra_tokens)
        vocab.add_to_vocab(traindata.corpus())
        if args.vocab_size is not None:
            vocab.cut_vocab(max_tokens=args.vocab_size)
        vocab.save("vocab_{}".format(len(vocab)))
        pad_token_id = vocab.tok2ind[PAD_TOKEN]
        start_token_id = vocab.tok2ind[START_TOKEN]

        encoder_opts = {
            "input_size": len(vocab),
            "embedding_size": args.embedding_size,
            "hidden_size": args.hidden_size,
            "aggregate_method": args.aggregate_method
        }
        decoder_opts = {
            "input_size": len(vocab),
            "embedding_size": args.embedding_size,
            "hidden_size": {
                "mean": args.embedding_size,
                "lstm": args.hidden_size,
                "bilstm": args.hidden_size * 2,
                "poolbilstm": args.hidden_size * 2            
            }[args.encoder],
            "output_size": len(vocab)
        }
        model = PersonaExtractor(args.encoder, encoder_opts, args.decoder, decoder_opts, start_token=start_token_id)
        criterion = nn.NLLLoss(ignore_index=pad_token_id, reduction='mean')

    if args.device == "mps":
        assert torch.backends.mps.is_available(), "Device 'mps' not available"
        assert torch.backends.mps.is_built(), "PyTorch installation was not built with MPS activated"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available"

    # wandb.init(project="pex", entity="thegist")
    # wandb.config.update(args)  # Converts args to a dictionary and logs it in wandb

    train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, collate_fn=traindata.batchify)
    valid_loader = torch.utils.data.DataLoader(dataset=validdata, batch_size=args.batch_size, shuffle=False, collate_fn=validdata.batchify)
    test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=False, collate_fn=testdata.batchify)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        print("Loading model from ", loadpath)
        model.load_state_dict(torch.load(loadpath))

    print("Start training")
    best_model, train_stats = train(
        model, train_loader, valid_loader, optimizer, criterion,
        device=args.device, epochs=args.epochs, log_interval=args.log_interval
    )

    if args.save != "":
        savepath = args.checkpoint_dir + savename(args)
        print("Saving model to ", savepath)
        torch.save(best_model.state_dict(), savepath)

    print("Start testing")
    test_stats = eval(model, test_loader, criterion, device=args.device)
    print("Test stats: ", test_stats)

