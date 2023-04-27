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
import json

import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import PrefixBert
from models.bart_extractor import PrefixBart, BartExtractor, ConditionalFactLoss, BART_BASE
from models.dialogpt import DialoGPT
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss
from models.knowledge_grounded_generator.kg_utils import ConceptGraph
from dataset.msc_kg_sessions import KG_enriched_MSC_Session
from dataset.msc_sessions import MSC_Session
from dataset.convai2 import ConvAI2
from dataset.msc_summary_turns import MSC_Turns
from dataset.tokenizer import train_tokenizer, Tokenizer, UNK_TOKEN, END_TOKEN, PAD_TOKEN
from utils.general import savename
import utils.logging as logging


def train(model, trainloader, validloader, optimizer, criterion, 
    device, epochs, log_interval, valid_interval, patience,
    do_grid_search, use_wandb):

    train_losses = []
    max_accuracy = -1
    min_loss = float('inf')
    step = 0
    model.to(device)
    best_model = model
    num_batches = len(trainloader)
    if patience is None:
        patience = num_batches
    patience_count = patience * epochs

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
                logging.info("Epoch {}, step {}: Train loss={:.4f}".format(epoch, step, loss_avg))    
    
            if (step % valid_interval == 0) or (step % num_batches == 0):
                # Evaluate on validation set
                model.eval()
                valid_stats = eval(model, validloader, criterion, device)
                valid_acc = valid_stats['valid_acc']
                valid_loss = valid_stats['valid_loss']
                logging.info("Epoch {}, step {}: Validation stats={}".format(epoch, step, valid_stats))
                patience_count -= 1
                model.train()

                if use_wandb:
                    valid_stats["epoch"] = epoch
                    wandb.log(valid_stats, step=step)

                if do_grid_search:
                    tune.report(valid_acc=valid_acc, valid_loss=valid_loss)
                    # tune.report(valid_loss=valid_loss)

                if valid_acc > max_accuracy:
                        max_accuracy = valid_acc
                        logging.info("Best accuracy improved to {:.2%}".format(max_accuracy))
                        patience_count = patience
                        best_model = copy.deepcopy(model)
                # if valid_loss < min_loss:
                #         min_loss = valid_loss
                #         logging.info("Best loss improved to {:.4f}".format(min_loss))
                #         patience_count = patience
                #         best_model = copy.deepcopy(model)
            if patience_count < 0:
                logging.info("Training loop terminated because it ran out of patience after {} validation interval(s) without improvement".format(patience + 1))
                break
        if patience_count < 0: 
            logging.info("Training loop terminated after epoch {}, step {}".format(epoch, step))
            break

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

    logging.info("Set up {} to {}".format(args.model, args.task))

    if args.task == 'classify': 
        
        # Classify whether dialog turns contain a fact

        if args.model == "bert":

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
                if (args.freeze is not None) and (num_added_toks > 0):
                    logging.warning("Added tokens {} are not trained, because part of model parameters is frozen (freeze={})".format(args.add_tokens, args.freeze))
            model = PrefixBert('bert-base-uncased', freeze=args.freeze, prefix_size=args.prefix_size, prefix_aggr=args.prefix_aggr)
            model.bert.resize_token_embeddings(len(tokenizer))
            criterion = nn.NLLLoss(reduction='mean')

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'session': args.session,
            'tokenizer': tokenizer,
            'len_context': args.len_context,
            'speaker_prefixes': args.speaker_prefixes,
            'nofact_token': args.nofact_token,
            'batch_format': 'huggingface',
            'batch_pad_id': tokenizer.pad_token_id
        } 
        with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")):
            traindata = MSC_Turn_Facts(subset='train', max_samples=args.train_samples, **dataset_config)
            validdata = MSC_Turn_Facts(subset='valid', max_samples=args.valid_samples, **dataset_config)
            testdata = MSC_Turn_Facts(subset='test', max_samples=args.test_samples, **dataset_config)
        collate_fn = traindata.batchify

    elif args.task == 'generate': 

        # Generate the fact(s) that are implied by the dialog turns (if any)

        if args.model == "seq2seq":

            if args.load == '':
                tokenizer = train_tokenizer(
                    corpus=MSC_Turns(basedir=args.basedir, session=args.session, subset='train', max_samples=args.train_samples),
                    max_size=args.vocab_size
                )
                if args.add_tokens is not None:
                    num_added_toks = tokenizer.add_tokens(args.add_tokens)
            else:
                tokenizer = Tokenizer.from_pretrained(args.checkpoint_dir + savename(args) + '_tokenizer.json')
            if args.save != '':
                tokenizer.save(args.checkpoint_dir + savename(args) + '_tokenizer')
            pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
            eos_token_id = tokenizer.token_to_id(END_TOKEN)
            unk_token_id = tokenizer.token_to_id(UNK_TOKEN)
            nofact_token_id = tokenizer.token_to_id(args.nofact_token) if args.nofact_token != '' else eos_token_id
            assert nofact_token_id != unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)
            vocab_size = tokenizer.get_vocab_size()
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
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            pad_token_id = tokenizer.pad_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
            assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

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
            model.bart.resize_token_embeddings(len(tokenizer))
            criterion = ConditionalFactLoss(nofact_token_id=nofact_token_id, ignore_index=tokenizer.pad_token_id, lm_weight=args.lm_loss_factor)

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        MSC_Turns.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'sessions': args.sessions
        } 
        with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
            traindata = MSC_Turns(subset='train', max_samples=args.train_samples, **dataset_config)
            validdata = MSC_Turns(subset='valid', max_samples=args.valid_samples, **dataset_config)
            testdata = MSC_Turns(subset='test', max_samples=args.test_samples, **dataset_config)
        collate_fn = partial(MSC_Turns.batchify, batch_format=model.batch_format, batch_pad_id=pad_token_id)


    elif args.task == "dialog": # Generate next utterance based on previous dialog turns

        if args.model == "kg_gen":
        
            tokenizer = AutoTokenizer.from_pretrained(args.lm)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if args.add_tokens is not None:
                tokenizer.add_tokens(args.add_tokens)
            tokenizer.bos_token_id = tokenizer.eos_token_id
            model = KnowledgeGroundedDecoder(vars(args), tokenizer, config=PretrainedConfig())
            model.gpt2model.resize_token_embeddings(len(tokenizer))
            criterion = KG_loss(ignore_index=tokenizer.pad_token_id, invalid=-1, alpha = args.alpha, beta = args.beta)

            kg = ConceptGraph(args.kg_datadir, args.kg)
            kg.build_reduced_graph(args.kg_datadir + args.dataset_concepts)
            del args.kg  # argument kg becomes the graph (instead of the filename)

        elif args.model == "dialogpt":

            tokenizer = AutoTokenizer.from_pretrained(args.lm)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if args.add_tokens is not None:
                tokenizer.add_tokens(args.add_tokens)
            tokenizer.bos_token_id = tokenizer.eos_token_id
            if args.speaker_prefixes is not None:
                tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(args.speaker_prefixes[0])
            model = DialoGPT(args.lm, tokenizer.bos_token_id)
            model.model.resize_token_embeddings(len(tokenizer))
            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        if args.session == 1:
            args.session = '-'.join(['1'] + args.convai2_version)
        dataset_config = vars(args)
        dataset_config.update({
            'basedir': args.datadir + args.basedir,
            'session': args.session,
            'tokenizer': tokenizer,
        })
        if args.model == "kg_gen":

            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                traindata = KG_enriched_MSC_Session(subset='train', kg=kg, max_samples=args.train_samples, **dataset_config)
                validdata = KG_enriched_MSC_Session(subset='valid', kg=kg, max_samples=args.valid_samples, **dataset_config)
                testdata = KG_enriched_MSC_Session(subset='test', kg=kg, max_samples=args.test_samples, **dataset_config)
            collate_fn = partial(traindata.batchify, batch_format=KnowledgeGroundedDecoder.batch_format, batch_pad_id=tokenizer.pad_token_id)

        elif args.model == "dialogpt":

            if args.persona_selector is not None:

                # Load pretrained model to select generate (tokens for) persona sentences from a batch with input_ids
                loadpath = args.checkpoint_dir + args.persona_selector
                logging.info("Loading persona_selector from {}".format(loadpath))
                with open(loadpath + '.config', 'r') as f:
                    bart_config = json.loads(f.read())
                bart_tokenizer = AutoTokenizer.from_pretrained(bart_config['bart_base'])
                if bart_config['add_tokens'] is not None:
                    bart_tokenizer.add_tokens(bart_config['add_tokens'])
                bart_model = BartExtractor(bart_config['bart_base'], bart_config['nofact_token_id'])
                bart_model.bart.resize_token_embeddings(len(bart_tokenizer))
                bart_model.load_state_dict(torch.load(loadpath, map_location=torch.device(args.device)))

                # Configure MSC_Turns to predict persona sentences from a list of utterances
                MSC_Turns.set(
                    tokenizer=bart_tokenizer, 
                    len_context=2, 
                    speaker_prefixes=bart_config['speaker_prefixes'], 
                    nofact_token=bart_config['nofact_token_id']
                )
                dataset_config['persona_selector'] = partial(MSC_Turns.predict_from_utterances, model=bart_model, device=args.device)

            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                traindata = MSC_Session(subset='train', max_samples=args.train_samples, **dataset_config)
                validdata = MSC_Session(subset='valid', max_samples=args.valid_samples, **dataset_config)
                testdata = MSC_Session(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = partial(traindata.batchify, batch_format=DialoGPT.batch_format, batch_pad_id=tokenizer.pad_token_id)
                
    if args.use_wandb:
        wandb.init(project="pex", entity="thegist")
        wandb.config.update(args)  # Converts args to a dictionary and logs it in wandb

    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        logging.info("Loading model from {}".format(loadpath))
        model.load_state_dict(torch.load(loadpath))
        
    train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(dataset=validdata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.valid_interval is None:
        args.valid_interval = len(train_loader)

    logging.info("Use train/valid/test dataset with {}/{}/{} samples".format(len(traindata), len(validdata), len(testdata)))
    logging.info("Start training")

    best_model, train_stats = train(
        model, train_loader, valid_loader, optimizer, criterion, 
        device=args.device, epochs=args.epochs, log_interval=args.log_interval, valid_interval=args.valid_interval, patience=args.patience,
        do_grid_search=args.do_grid_search, use_wandb=args.use_wandb
    )

    if not args.do_grid_search:

        if args.save != "":
            savepath = args.checkpoint_dir + savename(args)
            logging.info("Saving model to {}".format(savepath))
            torch.save(best_model.state_dict(), savepath)

        logging.info("Start testing")
        test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)    
        test_stats = eval(best_model, test_loader, criterion, device=args.device)
        logging.success("Test stats: {}".format(test_stats))
        train_stats["test_loss"] = test_stats["valid_loss"]
        train_stats["test_acc"] = test_stats["valid_acc"]
        if args.use_wandb:
            wandb.run.summary["test_accuracy"] = test_stats["valid_acc"]

        # logging.info("Reloading model from {}".format(savepath))
        # model.load_state_dict(torch.load(savepath))
        if args.task == "classify":
            eval_kwargs = {'device': args.device}
        elif args.task == "generate":
            if args.device == 'mps':
                args.device = 'cpu'
                logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
            eval_kwargs = {'device': args.device, 'decoder_max': args.decoder_max}
        elif args.task == "dialog":
            if args.device == 'mps':
                args.device = 'cpu'
                logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
            eval_kwargs = {'device': args.device, 'decoder_max': args.decoder_max, 'batch_size': 4}
            if args.model == "dialogpt":
                testdata.batch_format = "huggingface_x"

        logging.info("Evaluating model on {} samples of testdata in {} with arguments {}".format(len(testdata), args.basedir, eval_kwargs))
        eval_stats = testdata.evaluate(best_model, **eval_kwargs)
        report = '\n'.join(["{:<10}: {}".format(k, v) for k, v in eval_stats.items()])
        logging.report(report)

    return train_stats


def get_parser():

    parser = argparse.ArgumentParser(description="Train a model", conflict_handler="resolve")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--log_interval", type=int, default=10, help="report interval")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())    
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--load", type=str, default="", help="filename of model to load")
    parser.add_argument("--save", type=str, default="", help="filename to save the model")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--do_grid_search", default=False, action='store_true')
    parser.add_argument("--use_wandb", default=False, action='store_true')
    
    # Encoder and decoder model
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert", "bart", "prefixbart", "kg_gen", "dialogpt"], help="Model")

    # Task and Dataset
    parser.add_argument("--task", type=str, default="classify", choices=["generate", "classify", "dialog"])
    parser.add_argument("--datadir", type=str, default="./data/", help="Datadir")
    parser.add_argument("--basedir", type=str, default="msc/msc_personasummary/", help="Base directory for dataset")
    parser.add_argument("--train_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--valid_samples", type=int, default=None, help="Max number of test samples")
    parser.add_argument("--test_samples", type=int, default=None, help="Max number of test samples")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--valid_interval", type=int, default=None, help="validation interval")
    parser.add_argument("--patience", type=int, default=None, help="Number of validation intervals without improvement after which training will be terminated")

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
        "dialogpt": DialoGPT,
    }[args.model].add_cmdline_args(parser)

    if args.model == "seq2seq":
        parser.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")

    # Add cmdline arguments for task/dataset
    if args.task == "classify":
        parser = MSC_Turn_Facts.add_cmdline_args(parser)
    elif args.task == "generate":
        parser = MSC_Turns.add_cmdline_args(parser)
    elif args.task == "dialog": 
        if args.model == "kg_gen":
            parser = KG_enriched_MSC_Session.add_cmdline_args(parser)
        elif args.model == "dialogpt":
            parser = MSC_Session.add_cmdline_args(parser)
        args = parser.parse_known_args()[0]
        if args.session == 1:
            parser = ConvAI2.add_cmdline_args(parser)
        
    args = parser.parse_args()

    # Prepare logging
    logging.set_log_level(args.loglevel)
    if args.logdir is not None:
        logging.add_file_handler(logdir=args.logdir)
    logging.info("Args: {}".format('\n'.join(["{:20s}: {}".format(k, v) for k, v in vars(args).items()])))

    if args.do_grid_search:
        ray.init(
            configure_logging=True,
            logging_level="warning",
            )
        search_space = {
            "seed": tune.grid_search([42, 123, 2206]),
            # "prefix_aggr": tune.grid_search(["concat", "max", "avg"]),
            "speaker_prefixes": tune.grid_search([None, ["<self>", "<other>"]]),
            "nofact_token": tune.sample_from(lambda spec: "" if spec.config.speaker_prefixes is None else "<nofact>"),
            "add_tokens": tune.sample_from(
                lambda spec: 
                    spec.config.speaker_prefixes 
                    if spec.config.speaker_prefixes is None 
                    else spec.config.speaker_prefixes + [spec.config.nofact_token]
                ),
            # "learning_rate": tune.grid_search([1e-5, 1e-4, 1e-3]),
            # "batch_size": tune.grid_search([16, 32, 64]),
            # "prefix_size": tune.grid_search([0, 5]),
            # # If there is a prefix, then freeze all Bert layers
            # # If the is no prefix, then vary the number of frozen layers
            # "freeze": tune.sample_from(
            #         lambda spec: {
            #             0: None, 
            #             1: 8, 
            #             2: 12
            #         }[random.randint(0,2)]
            #         if spec.config.prefix_size == 0 else 12
            #     ),
        }
        trainable_with_resources = tune.with_resources(partial(train_with_args, args=args), {"gpu": 1})
        tuner = tune.Tuner(
            trainable=trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=HyperBandScheduler(),
                metric="valid_acc", 
                mode="max",
                num_samples=2,
                max_concurrent_trials=8
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


