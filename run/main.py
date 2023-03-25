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
from transformers import AutoTokenizer, PretrainedConfig
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import BertClassifier, PrefixBert
from models.bart_extractor import PrefixBart, BartExtractor, ConditionalFactLoss, BART_BASE
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss
from dataset.msc_kg_sessions import KG_enriched_MSC_Session
from dataset.msc_summary import MSC_Turns, PERSONA_TOKENS, NO_FACT_TOKEN
from dataset.tokenizer import train_tokenizer, UNK_TOKEN, END_TOKEN, PAD_TOKEN, START_TOKEN
from utils.general import savename
import utils.logging as logging
from run.eval import eval_bart_text, eval_gen_text



def train(model, trainloader, validloader, optimizer, criterion, 
    device, epochs, log_interval, valid_interval,
    do_grid_search, use_wandb):

    train_losses = []
    max_accuracy = -1
    step = 0
    model.to(device)
    best_model = model
    num_batches = len(trainloader)

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
    
            if (step % valid_interval == 0) or (step % num_batches == 0):
                # Evaluate on validation set
                model.eval()
                valid_stats = eval(model, validloader, criterion, device)
                valid_acc = valid_stats['valid_acc']
                logging.info("Epoch {}, step {}: Validation stats={}".format(epoch, step, valid_stats))
                model.train()

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
            traindata = MSC_Turn_Facts(args.datadir + args.traindata, tokenizer, len_context=2, persona_identifier=args.persona_identifier, max_samples=args.train_samples)
            validdata = MSC_Turn_Facts(args.datadir + args.validdata, tokenizer, len_context=2, persona_identifier=args.persona_identifier, max_samples=args.test_samples)
            testdata = MSC_Turn_Facts(args.datadir + args.testdata, tokenizer, len_context=2, persona_identifier=args.persona_identifier, max_samples=args.test_samples)
        if args.persona_identifier == "token":
            num_added_toks = tokenizer.add_tokens(PERSONA_TOKENS)
        model = PrefixBert('bert-base-uncased', freeze=args.freeze, prefix_size=args.prefix_size, prefix_aggr=args.prefix_aggr)
        model.bert.resize_token_embeddings(len(tokenizer))
        criterion = nn.NLLLoss(reduction='mean')

    elif args.task == 'generate':

        logging.info("Set up {} to {}".format(args.model, args.task))

        if args.model == "seq2seq":
            tokenizer = train_tokenizer(
                corpus=MSC_Turns(
                    args.datadir + args.traindata, 
                    tokenizer=None, 
                    len_context=2, 
                    persona_identifier=args.persona_identifier, 
                    max_samples=args.train_samples
                ),
                max_size=args.vocab_size
            )
            if args.persona_identifier == "token":
                tokenizer.add_special_tokens(PERSONA_TOKENS + [NO_FACT_TOKEN])
            else:
                assert tokenizer.token_to_id(NO_FACT_TOKEN) != tokenizer.token_to_id(UNK_TOKEN), "NO_FACT_TOKEN must be known token"
            pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
            eos_token_id = tokenizer.token_to_id(END_TOKEN)
            nofact_token_id = tokenizer.token_to_id(NO_FACT_TOKEN) if NO_FACT_TOKEN != '' else eos_token_id
            vocab_size = tokenizer.get_vocab_size()
            batch_format = "padded_sequences"
            encoder_opts = {
                "input_size": vocab_size,
                "embedding_size": args.embedding_size,
                "hidden_size": args.hidden_size,
                "aggregate_method": args.aggregate_method
            }
            decoder_opts = {
                "input_size": vocab_size,
                "embedding_size": args.embedding_size,
                "hidden_size": {
                    "mean": args.embedding_size,
                    "lstm": args.hidden_size,
                    "bilstm": args.hidden_size * 2,
                    "poolbilstm": args.hidden_size * 2            
                }[args.encoder],
                "output_size": vocab_size
            }
            model = PersonaExtractor(args.encoder, encoder_opts, args.decoder, decoder_opts, start_token=eos_token_id, nofact_token_id=nofact_token_id)
            criterion = nn.NLLLoss(ignore_index=pad_token_id)

        elif args.model[-4:] == "bart":

            tokenizer = AutoTokenizer.from_pretrained(BART_BASE)
            if args.persona_identifier == "token":
                tokenizer.add_special_tokens({'additional_special_tokens': PERSONA_TOKENS + [NO_FACT_TOKEN]})
            vocab_size = tokenizer.vocab_size
            pad_token_id = tokenizer.pad_token_id
            start_token_id = tokenizer.eos_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(NO_FACT_TOKEN)
            assert nofact_token_id != tokenizer.unk_token_id, "NO_FACT_TOKEN cannot be unknown token"

            if args.model == "bart":
                model = BartExtractor(bart_base=BART_BASE, nofact_token_id=nofact_token_id)
            else:
                model = PrefixBart(
                    bart_base=BART_BASE, 
                    nofact_token_id=nofact_token_id, 
                    freeze=args.freeze, 
                    enc_prefix_size=args.enc_prefix_size,
                    dec_prefix_size=args.dec_prefix_size,
                    prefix_aggr=args.prefix_aggr
                )
            criterion = ConditionalFactLoss(nofact_token_id=nofact_token_id, ignore_index=tokenizer.pad_token_id, lm_weight=args.lm_loss_factor)
            model.bart.resize_token_embeddings(len(tokenizer))

        with FileLock(os.path.expanduser(args.datadir + ".lock")): 
            traindata = MSC_Turns(args.datadir + args.traindata, tokenizer, 
                len_context=2, persona_identifier=args.persona_identifier, max_samples=args.train_samples,
                batch_format=batch_format, batch_pad_id=pad_token_id
            )
            validdata = MSC_Turns(args.datadir + args.validdata, tokenizer, 
                len_context=2, persona_identifier=args.persona_identifier, max_samples=args.valid_samples,
                batch_format=batch_format, batch_pad_id=pad_token_id
            )
            testdata = MSC_Turns(args.datadir + args.testdata, tokenizer, 
                len_context=2, persona_identifier=args.persona_identifier, max_samples=args.test_samples,
                batch_format=batch_format, batch_pad_id=pad_token_id
            )

    elif args.task == "dialog":
        logging.info("Set up {} to {}".format(args.model, args.task))
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
            traindata = KG_enriched_MSC_Session(vars(args), args.datadir + args.traindata, tokenizer, max_samples=args.train_samples, batch_pad_id=tokenizer.pad_token_id)
            validdata = KG_enriched_MSC_Session(vars(args), args.datadir + args.validdata, tokenizer, max_samples=args.test_samples, batch_pad_id=tokenizer.pad_token_id)
            testdata = KG_enriched_MSC_Session(vars(args), args.datadir + args.testdata, tokenizer, max_samples=args.test_samples, batch_pad_id=tokenizer.pad_token_id)
        model = KnowledgeGroundedDecoder(vars(args), tokenizer, config=PretrainedConfig())
        criterion = KG_loss(ignore_index=tokenizer.pad_token_id, invalid=-1, alpha = args.alpha, beta = args.beta)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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
    if args.valid_interval is None:
        args.valid_interval = len(train_loader)

    logging.info("Start training")
    best_model, train_stats = train(
        model, train_loader, valid_loader, optimizer, criterion, 
        device=args.device, epochs=args.epochs, log_interval=args.log_interval, valid_interval=args.valid_interval,
        do_grid_search=args.do_grid_search, use_wandb=args.use_wandb
    )

    if not args.do_grid_search:

        if args.save != "":
            savepath = args.checkpoint_dir + savename(args)
            logging.info("Saving model to {}".format(savepath))
            torch.save(best_model.state_dict(), savepath)

        logging.info("Start testing")
        test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=False, collate_fn=testdata.batchify)    
        test_stats = eval(best_model, test_loader, criterion, device=args.device)
        logging.success("Test stats: {}".format(test_stats))
        train_stats["test_loss"] = test_stats["valid_loss"]
        train_stats["test_acc"] = test_stats["valid_acc"]
        if args.use_wandb:
            wandb.run.summary["test_accuracy"] = test_stats["valid_acc"]

        logging.info("Reloading model from {}".format(savepath))
        model.load_state_dict(torch.load(savepath))
        logging.info("Testing model with {}".format(args.testdata))
        if args.model[-4:] == "bart":
            eval_stats = eval_bart_text(model.to("cpu"), testdata, tokenizer, decoder_max=args.decoder_max)
        elif args.model == "kg_gen":
            eval_stats = eval_gen_text(model.to("cpu"), testdata, tokenizer, decoder_max=args.decoder_max, batch_size=4)
        report = '\n'.join(["{:<10}: {}".format(k, v) for k, v in eval_stats.items()])
        logging.report(report)

    return train_stats


def get_parser():

    parser = argparse.ArgumentParser(description="Train a model", conflict_handler="resolve")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--log_interval", type=int, default=10, help="report interval")
    parser.add_argument("--valid_interval", type=int, default=None, help="validation interval")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())    
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--load", type=str, default="", help="filename of model to load")
    parser.add_argument("--save", type=str, default="", help="filename to save the model")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--task", type=str, default="classify", choices=["generate", "classify", "dialog"])
    parser.add_argument("--do_grid_search", default=False, action='store_true')
    parser.add_argument("--use_wandb", default=False, action='store_true')
    
    # Encoder and decoder model
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert", "bart", "prefixbart", "kg_gen"], help="Model")

    # Dataset
    parser.add_argument("--datadir", type=str, default="./data/", help="Datadir")
    parser.add_argument("--traindata", type=str, default="msc/msc_personasummary/session_1/train.txt", help="Dataset file for training")
    parser.add_argument("--validdata", type=str, default="msc/msc_personasummary/session_1/valid.txt", help="Dataset file for validation")
    parser.add_argument("--testdata", type=str, default="msc/msc_personasummary/session_1/test.txt", help="Dataset file for testing")
    parser.add_argument("--train_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--valid_samples", type=int, default=None, help="Max number of test samples")
    parser.add_argument("--test_samples", type=int, default=None, help="Max number of test samples")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    return parser

if __name__ == "__main__":
    import argparse

    parser = get_parser()
    args = parser.parse_known_args()[0]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check availability of requested device
    if args.device == "mps":
        assert torch.backends.mps.is_available(), "Device 'mps' not available"
        assert torch.backends.mps.is_built(), "PyTorch installation was not built with MPS activated"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available"

    # Add cmdline arguments for model
    parser = {
        "seq2seq": PersonaExtractor,
        "bert": PrefixBert,
        "bart": BartExtractor,
        "prefixbart": PrefixBart,
        "kg_gen": KnowledgeGroundedDecoder,
    }[args.model].add_cmdline_args(parser)

    if args.model == "seq2seq":
        parser.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")

    # Add cmdline arguments for task/dataset
    parser = {
        "classify": MSC_Turn_Facts,
        "generate": MSC_Turns,
        "dialog": KG_enriched_MSC_Session,
    }[args.task].add_cmdline_args(parser)
    
    args = parser.parse_args()

    # Prepare logging
    logging.set_log_level(args.loglevel)
    if args.logdir is not None:
        logging.add_file_handler(logdir=args.logdir)
    logging.info("Args: {}".format(args))

    if args.do_grid_search:
        ray.init(
            configure_logging=True,
            logging_level="warning",
            )
        search_space = {
            "prefix_aggr": tune.grid_search(["concat", "max", "avg"]),
            "learning_rate": tune.grid_search([1e-4, 1e-3]),
            # "batch_size": tune.grid_search([16, 64]),
            # "prefix_size": tune.grid_search([0, 5]),
            # "freeze": tune.sample_from(lambda spec: {0:None, 1:8, 2:12}[random.randint(0,2)] if spec.config.prefix_size == 0 else 12),
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
        logging.success("BEST METRICS: {:.2%}".format(best_result.metrics["valid_acc"]))

    else:
        stats = train_with_args(config=None, args=args)


