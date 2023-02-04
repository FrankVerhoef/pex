"""
python run/main.py
to run with defaults
"""
import torch
import torch.nn as nn
from torch import optim
import random
import wandb

from models.persona_extractor import PersonaExtractor
from dataset.msc_summary import MSC_Turns, extra_tokens
from dataset.vocab import Vocab, PAD_TOKEN, START_TOKEN

def train_step(batch, model, optimizer, criterion, device):

    xs, ys, xs_len, ys_len = batch
    xs = xs.to(device)
    ys = ys.to(device)
    optimizer.zero_grad()
    
    output = model(xs, xs_len, teacher_forcing=True, ys=ys)
    loss = criterion(output.transpose(1,2), ys)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(model, dataloader, optimizer, criterion, device="cpu", epochs=1, log_interval=1000):

    losses = []
    model.to(device)

    for epoch in range(epochs):
        for step, batch in enumerate(iter(dataloader)):

            loss = train_step(batch, model, optimizer, criterion, device)
            losses.append(loss)

            if (step + 1) % log_interval == 0:
                loss_avg = sum(losses[-log_interval:]) / log_interval
                wandb.log({
                    "train_loss": loss_avg
                })
                print("Epoch {}, step {}: loss={}".format(epoch, step+1, loss_avg))
    
    return model, {"train_loss": losses}

def test(model, dataloader, criterion, device):

    losses = []
    total_correct = 0
    num_tokens = 0
    model.to(device)
    
    for batch in iter(dataloader):

        xs, ys, xs_len, ys_len = batch
        xs = xs.to(device)
        ys = ys.to(device)

        with torch.no_grad():
            output = model(xs, xs_len, max=ys.size(1))
            loss = criterion(output.transpose(1,2), ys)
            pred = output.argmax(dim=-1)

        ys, pred = ys.cpu(), pred.cpu()
        total_correct += pred.where(ys != criterion.ignore_index, torch.tensor(-1)).eq(ys.view_as(pred)).sum().item()
        num_tokens += sum(ys_len)
        losses.append(loss.item())

    test_stats = {
        "test_loss": sum(losses) / len(losses),
        "test_acc": total_correct / num_tokens
    }
    wandb.run.summary.update(test_stats)

    return test_stats

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
    
    # Encoder and decoder model
    parser.add_argument("--encoder", type=str, default="mean", help="Encoder model")
    parser.add_argument("--embedding_size", type=int, default=16, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size")
    parser.add_argument("--aggregate_method", type=str, default="avg", choices=["avg", "max"], help="Aggregate method for Pool Bi-LSTM")
    parser.add_argument("--decoder", type=str, default="lstm", help="Decoder model")
    parser.add_argument("--decoder_max", type=int, default=20, help="Max number of tokens to generate with decoder")

    # Dataset
    parser.add_argument("--datadir", type=str, default="/Users/FrankVerhoef/Programming/PEX/data/", help="Datadir")
    parser.add_argument("--traindata", type=str, default="msc/msc_personasummary/session_1/train.txt", help="Dataset file for training")
    parser.add_argument("--testdata", type=str, default="msc/msc_personasummary/session_1/test.txt", help="Dataset file for testing")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Max number of unique token (excluding special tokens)")
    parser.add_argument("--train_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--test_samples", type=int, default=None, help="Max number of training samples")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    random.seed(args.seed)

    vocab = Vocab()
    traindata = MSC_Turns(args.datadir + args.traindata, vocab.text2vec, len_context=2, max_samples=args.train_samples)
    testdata = MSC_Turns(args.datadir + args.testdata, vocab.text2vec, len_context=2, max_samples=args.test_samples)
    vocab.add_special_tokens(extra_tokens)
    vocab.add_to_vocab(traindata.corpus())
    vocab.cut_vocab(max_tokens=args.vocab_size)

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
    model = PersonaExtractor(args.encoder, encoder_opts, args.decoder, decoder_opts, start_token=vocab.tok2ind[START_TOKEN])

    if args.device == "mps":
        assert torch.backends.mps.is_available(), "Device 'mps' not available"
        assert torch.backends.mps.is_built(), "PyTorch installation was not built with MPS activated"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available"

    wandb.init(project="pex", entity="thegist")
    wandb.config.update(args)  # Converts args to a dictionary and logs it in wandb

    train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, collate_fn=traindata.batchify)
    test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=True, collate_fn=testdata.batchify)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss(ignore_index=vocab.tok2ind[PAD_TOKEN], reduction='mean')

    best_model, train_stats = train(
        model, train_loader, optimizer, criterion,
        device=args.device, epochs=args.epochs, log_interval=args.log_interval
    )

    test_stats = test(
        model, test_loader, criterion,
        device=args.device
    )
    print("Test stats: ", test_stats)

