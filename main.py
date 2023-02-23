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
import os
from functools import partial
from filelock import FileLock

import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

import transformers
from transformers import AutoTokenizer
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import BertClassifier, FrozenBert, PrefixBert
from dataset.msc_summary import MSC_Turns, extra_tokens
from dataset.vocab import Vocab, PAD_TOKEN, START_TOKEN
from utils.general import savename
import utils.logging as logging



def train(model, trainloader, validloader, optimizer, criterion, 
    device, epochs, log_interval,
    do_grid_search, use_wandb):

    train_losses = []
    max_accuracy = 0
    step = 0
    model.to(device)

    for epoch in range(epochs):

        # Train for one epoch
        model.train()
        for batch in iter(trainloader):
            step += 1
            loss = model.train_step(batch, optimizer, criterion, device)
            train_losses.append(loss)
    
            if step % log_interval == 0:
                loss_avg = sum(train_losses[-log_interval:]) / log_interval
                if use_wandb:
                    wandb.log({"train_loss": loss_avg, "epoch": epoch}, step=step)
                logging.verbose("Epoch {}, step {}: Train loss={:.4f}".format(epoch, step, loss_avg))
    
        # Evaluate on validation set
        model.eval()
        valid_stats = eval(model, validloader, criterion, device)
        valid_acc = valid_stats['valid_acc']
        logging.info("Epoch {}, step {}: Validation stats={}".format(epoch, step, valid_stats))

        if use_wandb:
            valid_stats["epoch"] = epoch
            wandb.log(valid_stats, step=step)

        if do_grid_search:
            tune.report(valid_acc=valid_acc)

        if valid_acc > max_accuracy:
                max_accuracy = valid_acc
                logging.info("Best accuracy improved to {:.2%}".format(max_accuracy))
                best_model = copy.deepcopy(model)

    return best_model, {"valid_acc": max_accuracy}


def eval(model, dataloader, criterion, device):

    eval_stats = []
    model.to(device)
    model.eval()

    for batch in iter(dataloader):

        stats = model.valid_step(batch, criterion, device)
        eval_stats.append(stats)

    stats = {
        "valid_loss": sum([stats["loss"] for stats in eval_stats]) / len(eval_stats),
        "valid_acc": sum([stats["acc"] for stats in eval_stats]) / len(eval_stats)}

    return stats


def train_with_args(config, args):

    if config is not None:
        for k, v in config.items():
            logging.info("Override/set {} to {}".format(k, v))
            setattr(args, k, v)

    if args.task == 'classify':

        logging.info("Set up {} to {}".format(args.model, args.task))
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        with FileLock(os.path.expanduser(args.datadir + ".lock")): 
            traindata = MSC_Turn_Facts(args.datadir + args.traindata, tokenizer, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.train_samples)
            validdata = MSC_Turn_Facts(args.datadir + args.validdata, tokenizer, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.test_samples)
            testdata = MSC_Turn_Facts(args.datadir + args.testdata, tokenizer, len_context=2, persona_tokens=args.persona_tokens, max_samples=args.test_samples)
        if args.persona_tokens:
            num_added_toks = tokenizer.add_tokens(extra_tokens)
        if args.load == "":
            model = PrefixBert('bert-base-uncased', freeze=args.freeze, prefix_size=args.prefix_size)
        else:
            model = PrefixBert()
        model.bert.resize_token_embeddings(len(tokenizer))
        criterion = nn.NLLLoss(reduction='mean')

    elif args.task == 'generate':

        logging.info("Set up {} to {}".format(args.model, args.task))
        vocab = Vocab()
        with FileLock(os.path.expanduser(args.datadir + ".lock")): 
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

    if args.use_wandb:
        wandb.init(project="pex", entity="thegist")
        wandb.config.update(args)  # Converts args to a dictionary and logs it in wandb

    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        logging.info("Loading model from {}".format(loadpath))
        model.load_state_dict(torch.load(loadpath))
        
    train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, collate_fn=traindata.batchify)
    valid_loader = torch.utils.data.DataLoader(dataset=validdata, batch_size=args.batch_size, shuffle=False, collate_fn=validdata.batchify)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logging.info("Start training with config: {}".format("None" if config is None else config))
    best_model, train_stats = train(
        model, train_loader, valid_loader, optimizer, criterion, 
        device=args.device, epochs=args.epochs, log_interval=args.log_interval,
        do_grid_search=args.do_grid_search, use_wandb=args.use_wandb
    )

    if not args.do_grid_search:

        if args.save != "":
            savepath = args.checkpoint_dir + savename(args)
            logging.info("Saving model to {}".format(savepath))
            torch.save(best_model.state_dict(), savepath)

        test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=False, collate_fn=testdata.batchify)    
        logging.info("Start testing")
        test_stats = eval(best_model, test_loader, criterion, device=args.device)
        logging.success("Test stats: {}".format(test_stats))
        train_stats["test_loss"] = test_stats["valid_loss"]
        train_stats["test_acc"] = test_stats["valid_acc"]
        if args.use_wandb:
            wandb.run.summary["test_accuracy"] = test_stats["valid_acc"]

    return train_stats


def parse_args():

    parser = argparse.ArgumentParser(description="Train a model")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--log_interval", type=int, default=10, help="report interval")
    parser.add_argument("--loglevel", type=str, default="info", choices=logging.get_all_levels())    
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--load", type=str, default="", help="filename of model to load")
    parser.add_argument("--save", type=str, default="", help="filename to save the model")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--task", type=str, default="classify", choices=["generate", "classify"])
    parser.add_argument("--do_grid_search", default=False, action='store_true')
    parser.add_argument("--use_wandb", default=False, action='store_true')
    
    # Encoder and decoder model
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert"], help="Encoder model")
    parser.add_argument("--encoder", type=str, default="mean", help="Encoder model")
    parser.add_argument("--embedding_size", type=int, default=100, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--aggregate_method", type=str, default="avg", choices=["avg", "max"], help="Aggregate method for Pool Bi-LSTM")
    parser.add_argument("--decoder", type=str, default="lstm", help="Decoder model")
    parser.add_argument("--decoder_max", type=int, default=20, help="Max number of tokens to generate with decoder")

    # Bert variants
    parser.add_argument("--freeze", type=int, default=0, help="Layers to freeze for finetuning; None=none, 0=only embeddings, 12=all")
    parser.add_argument("--prefix_size", type=int, default=0, help="Insert prefix in BERT")

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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    return parser.parse_args()

if __name__ == "__main__":
    import argparse

    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.set_log_level(args.loglevel)
    if args.logdir is not None:
        logging.add_file_handler(logdir=args.logdir)
    logging.info("Args: {}".format(args))

    if args.device == "mps":
        assert torch.backends.mps.is_available(), "Device 'mps' not available"
        assert torch.backends.mps.is_built(), "PyTorch installation was not built with MPS activated"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available"

    if args.do_grid_search:
        ray.init(
            configure_logging=True,
            logging_level="warning",
            )
        search_space = {
            "learning_rate": tune.grid_search([1e-4, 1e-3]),
            "batch_size": tune.grid_search([16, 64]),
            "prefix_size": tune.grid_search([0, 5]),
            "freeze": tune.sample_from(lambda spec: {0:None, 1:8, 2:12}[random.randint(0,2)] if spec.config.prefix_size == 0 else 12),
        }
        tuner = tune.Tuner(
            trainable=partial(train_with_args, args=args),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=HyperBandScheduler(),
                metric="valid_acc", 
                mode="max",
                num_samples=1,
                max_concurrent_trials=4
            ),
            run_config = air.RunConfig(
                verbose=3,
            )
        )
        results = tuner.fit()

        best_result = results.get_best_result() 
        logging.success("BEST RESULTS: {}".format(best_result.config))
        logging.success("BEST METRICS: {.2%}".format(best_result.metrics["valid_acc"]))

    else:
        stats = train_with_args(config=None, args=args)


