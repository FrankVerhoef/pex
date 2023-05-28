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
import configargparse as argparse

from transformers import AutoTokenizer, PretrainedConfig
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import PrefixBert
from models.bart_extractor import PrefixBart, BartExtractor, ConditionalFactLoss, ExtractedFactLoss
from models.dialogpt import DialoGPT
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss
from models.knowledge_grounded_generator.kg_utils import ConceptGraph
from dataset.msc_kg_sessions import KG_enriched_MSC_Session
from dataset.msc_sessions import MSC_Session
from dataset.convai2 import ConvAI2
from dataset.msc_summary_turns import MSC_Turns
from dataset.tokenizer import train_tokenizer, Tokenizer, UNK_TOKEN, END_TOKEN, PAD_TOKEN
from metrics.terp import TerpMetric
from run.tune import do_tune
from ray.air import session, RunConfig
from ray.tune import with_resources
from utils.general import savename, prettydict, dict_with_key_prefix, save_config, save_dict
from utils.listdict import ListDict
import utils.logging as logging


def train(model, trainloader, validloader, optimizer, criterion, 
    device, epochs, log_interval, valid_interval, patience,
    do_tune, use_wandb):

    train_losses = []
    saved_stats = {"valid_loss": float('inf')}
    step = 0
    model.to(device)
    best_model = model
    num_batches = len(trainloader)
    if patience is None:
        patience = num_batches * epochs
    patience_count = patience
    total_original_tokens, total_truncated_tokens = 0, 0

    for epoch in range(epochs):

        # Train for one epoch
        model.train()
        for batch in iter(trainloader):
            step += 1
            if hasattr(batch, "num_truncated_tokens"):
                total_original_tokens += batch.num_original_tokens
                total_truncated_tokens += batch.num_truncated_tokens

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
                valid_stats = valid(model, validloader, criterion, device)
                valid_stats = dict_with_key_prefix(valid_stats, prefix="valid_")
                logging.info("Epoch {}, step {}: Validation stats={}".format(epoch, step, valid_stats))
                patience_count -= 1
                model.train()

                if use_wandb:
                    valid_stats["epoch"] = epoch
                    wandb.log(valid_stats, step=step)

                if do_tune:
                    session.report(valid_stats)

                if valid_stats["valid_loss"] < saved_stats["valid_loss"]:
                        saved_stats = valid_stats
                        logging.info("Best loss improved to {:.4f}".format(saved_stats["valid_loss"]))
                        patience_count = patience
                        best_model = copy.deepcopy(model)

            if patience_count < 0:
                logging.info("Training loop terminated because it ran out of patience after {} validation interval(s) without improvement".format(patience + 1))
                break
        if patience_count < 0: 
            logging.info("Training loop terminated after epoch {}, step {}".format(epoch, step))
            break
    logging.info("Average truncation: {}".format(total_truncated_tokens / max(total_original_tokens, 1)))
    return best_model, saved_stats


def valid(model, dataloader, criterion, device):

    valid_stats = ListDict()
    model.to(device)
    model.eval()

    for batch in iter(dataloader):

        stats = model.valid_step(batch, criterion, device)
        valid_stats.append(stats)

    stats = valid_stats.mean()

    return stats

def evaluate(model, testdata, args):

    if args.task == "classify":
        eval_kwargs = {'device': args.device}
    elif args.task == "generate":
        TerpMetric.set(terp_dir=args.terpdir, java_home=args.java_home, tmp_dir=args.tmpdir)
        if args.device == 'mps':
            args.device = 'cpu'
            logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
        eval_kwargs = {'device': args.device, 'decoder_max': args.decoder_max, 'batch_size': 4}
    elif args.task == "dialog":
        if args.device == 'mps':
            args.device = 'cpu'
            logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
        eval_kwargs = {'device': args.device, 'decoder_max': args.decoder_max, 'batch_size': 4}
        if args.model == "dialogpt":
            testdata.batch_format = "huggingface_x"

    logging.info("Evaluating model on {} samples of testdata in {} with arguments {}".format(len(testdata), args.basedir, eval_kwargs))
    eval_stats = testdata.evaluate(model, **eval_kwargs)
    logging.report(prettydict(eval_stats, title="Eval_stats"))

    return eval_stats 


def prepare_model_and_data(args):

    model, traindata, validdata, testdata, collate_fn, criterion = None, None, None, None, None, None

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

        MSC_Turn_Facts.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'sessions': args.sessions
        }
        with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")):
            if args.action in ['tune', 'train']:
                traindata = MSC_Turn_Facts(subset='train', max_samples=args.train_samples, **dataset_config)
                validdata = MSC_Turn_Facts(subset='valid', max_samples=args.valid_samples, **dataset_config)
            if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                testdata = MSC_Turn_Facts(subset='test', max_samples=args.test_samples, **dataset_config)
        collate_fn = MSC_Turn_Facts.batchify

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

            tokenizer = AutoTokenizer.from_pretrained(args.bart_base)
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            pad_token_id = tokenizer.pad_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
            assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

            if args.model == "bart":
                model = BartExtractor(bart_base=args.bart_base, nofact_token_id=nofact_token_id)
            else:
                model = PrefixBart(
                    bart_base=args.bart_base, 
                    nofact_token_id=nofact_token_id, 
                    freeze=args.freeze, 
                    enc_prefix_size=args.enc_prefix_size,
                    dec_prefix_size=args.dec_prefix_size,
                    prefix_aggr=args.prefix_aggr
                )
            model.bart.resize_token_embeddings(len(tokenizer))
            criterion = ExtractedFactLoss(nofact_token_id=nofact_token_id, ignore_index=tokenizer.pad_token_id, lm_weight=args.lm_loss_factor, clf_loss=args.clf_loss)

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        MSC_Turns.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'sessions': args.sessions
        } 
        with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
            if args.action in ['tune', 'train']:
                traindata = MSC_Turns(subset='train', max_samples=args.train_samples, **dataset_config)
                validdata = MSC_Turns(subset='valid', max_samples=args.valid_samples, **dataset_config)
            if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                testdata = MSC_Turns(subset='test', max_samples=args.test_samples, **dataset_config)
        collate_fn = partial(MSC_Turns.batchify, with_labels=True, batch_format=model.batch_format, batch_pad_id=pad_token_id)

    elif args.task == "dialog": # Generate next utterance based on previous dialog turns

        if args.model == "kg_gen":
        
            tokenizer = AutoTokenizer.from_pretrained(args.lm)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if args.add_tokens is not None:
                tokenizer.add_tokens(args.add_tokens)
            tokenizer.bos_token_id = tokenizer.eos_token_id
            model = KnowledgeGroundedDecoder(
                args.lm, tokenizer.bos_token_id, args.fixed_lm, args.num_hops, args.gamma, args.aggregate_method, args.block_src, args.gate,
                tokenizer, 
                config=PretrainedConfig()
            )
            model.gpt2model.resize_token_embeddings(len(tokenizer))
            criterion = KG_loss(ignore_index=tokenizer.pad_token_id, invalid=-1, alpha = args.alpha, beta = args.beta)

            kg = ConceptGraph(args.kg_datadir, args.kg)
            kg.build_reduced_graph(args.kg_datadir + args.dataset_concepts)

            KG_enriched_MSC_Session.set(
                tokenizer, args.speaker_prefixes, 
                kg, args.num_hops, args.max_branch, args.max_concepts, args.max_triples, args.overlapping_concepts
            )

            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'session': args.session if args.session != 1 else '-'.join(['1'] + args.convai2_version),
                'include_persona': args.include_persona
            }

            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                if args.action in ['tune', 'train']:
                    traindata = KG_enriched_MSC_Session(subset='train', max_samples=args.train_samples, **dataset_config)
                    validdata = KG_enriched_MSC_Session(subset='valid', max_samples=args.valid_samples, **dataset_config)
                if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                    testdata = KG_enriched_MSC_Session(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = partial(KG_enriched_MSC_Session.batchify, batch_format=KnowledgeGroundedDecoder.batch_format)

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

            MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=args.speaker_prefixes)

            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'session': args.session if args.session != 1 else '-'.join(['1'] + args.convai2_version),
                'include_persona': args.include_persona,
                'include_history': args.include_history
            }

            if args.persona_selector is not None:

                # Load pretrained model to select generate (tokens for) persona sentences from a batch with input_ids
                loadpath = args.checkpoint_dir + args.persona_selector
                logging.info("Loading persona_selector from {}".format(loadpath))
                with open(loadpath + '.config', 'r') as f:
                    bart_config = json.loads(f.read())
                bart_tokenizer = AutoTokenizer.from_pretrained(bart_config['bart_base'])
                if bart_config['add_tokens'] is not None:
                    bart_tokenizer.add_tokens(bart_config['add_tokens'])
                bart_nofact_token_id = tokenizer.convert_tokens_to_ids(bart_config['nofact_token']) if bart_config['nofact_token'] != '' else bart_tokenizer.eos_token_id
                bart_model = BartExtractor(bart_config['bart_base'], bart_nofact_token_id)
                bart_model.bart.resize_token_embeddings(len(bart_tokenizer))
                bart_device = args.device
                if bart_device == 'mps':
                    bart_device = 'cpu'
                    logging.warning("Changed device from 'mps' to 'cpu' for BART persona selector")
                bart_model.load_state_dict(torch.load(loadpath, map_location=torch.device(bart_device)))

                # Configure MSC_Turns to predict persona sentences from a list of utterances
                MSC_Turns.set(
                    tokenizer=bart_tokenizer, 
                    len_context=2, 
                    speaker_prefixes=bart_config['speaker_prefixes'], 
                    nofact_token=bart_nofact_token_id
                )
                dataset_config['persona_selector'] = partial(MSC_Turns.predict_from_utterances, model=bart_model, device=bart_device)

            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                if args.action in ['tune', 'train']:
                    traindata = MSC_Session(subset='train', max_samples=args.train_samples, augmented=args.augmented, **dataset_config)
                    validdata = MSC_Session(subset='valid', max_samples=args.valid_samples, augmented=args.augmented, **dataset_config)
                if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                    testdata = MSC_Session(subset='test', max_samples=args.test_samples, augmented=True, **dataset_config)
            collate_fn = partial(MSC_Session.batchify, with_labels=True, batch_format=DialoGPT.batch_format, batch_pad_id=tokenizer.pad_token_id, buffer=0)

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

    return model, traindata, validdata, testdata, collate_fn, criterion


def train_with_args(config, args):

    if config is not None:
        for k, v in config.items():
            logging.info("Override/set {} to {}".format(k, v))
            setattr(args, k, v)

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Set up for model: {}, task: {}".format(args.model, args.task))
    model, traindata, validdata, testdata, collate_fn, criterion = prepare_model_and_data(args)

    if args.use_wandb:
        wandb.init(project="pex", entity="thegist")
        wandb.config.update(args)  # Converts args to a dictionary and logs it in wandb

    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        logging.info("Loading model from {}".format(loadpath))
        model.load_state_dict(torch.load(loadpath, map_location=torch.device(args.device)))

    stats = {}
    if args.action in ['tune', 'train']: 
        train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(dataset=validdata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.valid_interval is None:
            args.valid_interval = len(train_loader)

        logging.info("Start training")
        logging.info("Use train/valid dataset with {}/{} samples".format(len(traindata), len(validdata)))
 
        model, valid_stats = train(
            model, train_loader, valid_loader, optimizer, criterion, 
            device=args.device, epochs=args.epochs, log_interval=args.log_interval, valid_interval=args.valid_interval, patience=args.patience,
            do_tune=args.action == 'tune', use_wandb=args.use_wandb
        )
        stats = valid_stats

        if args.action == 'train' and args.save != "":
            savepath = args.checkpoint_dir + savename(args)
            logging.info("Saving model to {}".format(savepath))
            torch.save(model.state_dict(), savepath)
            save_config(savepath + '.config', args)

    if args.action in ['train', 'eval'] and not args.skip_eval:

        logging.info("Start testing")
        test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)  
        test_stats = valid(model, test_loader, criterion, device=args.device)

        logging.report("Test stats: {}".format(test_stats))
        stats.update(dict_with_key_prefix(test_stats, prefix="test_"))

        if args.use_wandb:
            wandb.run.summary["test_accuracy"] = stats["test_acc"]  

        eval_stats = evaluate(model, testdata, args)
        stats.update(dict_with_key_prefix(eval_stats, prefix="eval_"))


    return stats


def get_args():

    parser = argparse.ArgumentParser(description="PERSONA EXTRACTOR (note: models and tasks have additional options, please consult the documentation)", conflict_handler="resolve")

    # General, loading, saving, logging
    generalgroup = parser.add_argument_group("general options and setting for loading, saving, monitoring")
    generalgroup.add_argument("--configfile", is_config_file=True, help="configfile with default value (will be overridden by cmdline arguments)")
    generalgroup.add_argument("--seed", type=int, default=42, help="random seed")
    generalgroup.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    generalgroup.add_argument("--output_dir", type=str, default="./output/")
    generalgroup.add_argument("--log_interval", type=int, default=10, help="report interval")
    generalgroup.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())    
    generalgroup.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    generalgroup.add_argument("--load", type=str, default="", help="filename of model to load")
    generalgroup.add_argument("--save", type=str, default="", help="filename to save the model")
    generalgroup.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    generalgroup.add_argument("--use_wandb", default=False, action='store_true')

    # Main arguments
    parser.add_argument("action", type=str, choices=['tune', 'train', 'eval'], help="choose an action")
    parser.add_argument("model", type=str, choices=["seq2seq", "bert", "bart", "prefixbart", "kg_gen", "dialogpt"], help="choose one of the available models")
    parser.add_argument("task", type=str, choices=["generate", "classify", "dialog"], help="choose a task/dataset to use for tuning/training/evaluation")

    tune_group = parser.add_argument_group("options for tuning")
    tune_group.add_argument("--experiment_name", type=str, default="trainpex", help="experiment name for Ray Tune")
    
    traingroup = parser.add_argument_group("options for training")
    traingroup.add_argument("--epochs", type=int, default=1, help="number of epochs for training")
    traingroup.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    traingroup.add_argument("--valid_interval", type=int, default=None, help="validation interval")
    traingroup.add_argument("--patience", type=int, default=None, help="number of validation intervals without improvement after which training will be terminated")
    traingroup.add_argument("--batch_size", type=int, default=32, help="batch size")
    traingroup.add_argument("--skip_eval", default=False, action='store_true', help="just train")

    evalgroup = parser.add_argument_group("options for evaluation")
    evalgroup.add_argument("--metrics", nargs='*', help="only report listed metrics")
    evalgroup.add_argument("--print_max", type=int, default=20, help="max number of test examples to print")

    args = parser.parse_known_args()[0]

    # Add cmdline arguments for model
    modelgroup = parser.add_argument_group("Options for the chosen model")
    {
        "seq2seq": PersonaExtractor,
        "bert": PrefixBert,
        "bart": BartExtractor,
        "prefixbart": PrefixBart,
        "kg_gen": KnowledgeGroundedDecoder,
        "dialogpt": DialoGPT,
    }[args.model].add_cmdline_args(modelgroup)

    if args.model == "seq2seq":
        modelgroup.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")

    # Add cmdline arguments for Task/Dataset
    parser.add_argument("--datadir", type=str, default="./data/", help="root directory for the dataset files")
    parser.add_argument("--basedir", type=str, default="msc/msc_personasummary/", help="base directory for dataset")
    parser.add_argument("--train_samples", type=int, default=None, help="max number of training samples")
    parser.add_argument("--valid_samples", type=int, default=None, help="max number of test samples")
    parser.add_argument("--test_samples", type=int, default=None, help="max number of test samples")

    if args.task == "classify":
        parser = MSC_Turn_Facts.add_cmdline_args(parser)
    elif args.task == "generate":
        if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
            print("CHECK: ", vars(args))
            parser = TerpMetric.add_cmdline_args(parser)
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

    return args


if __name__ == "__main__":

    args = get_args()

    # Check availability of requested device
    if args.device == "mps":
        assert torch.backends.mps.is_available(), "Device 'mps' not available"
        assert torch.backends.mps.is_built(), "PyTorch installation was not built with MPS activated"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available"

    # Prepare logging
    logging.set_log_level(args.loglevel)
    if args.logdir is not None:
        logging.add_file_handler(logdir=args.logdir)
    logging.info(prettydict(vars(args), title="Args"))

    if args.action == 'tune':
        ray_dir = args.output_dir + "ray_results"
        trainable = partial(train_with_args, args=args)
        if args.device == 'cuda':
            trainable = with_resources(trainable, {"gpu": 1})
        run_config = RunConfig(
            local_dir=ray_dir,
            name=args.experiment_name 
        )
        results = do_tune(train_fn=trainable, run_config=run_config)
        save_config(f"{ray_dir}/{args.experiment_name}/base.config", args)
        logging.info(f"Ray results saved in {ray_dir}/{args.experiment_name}")
    else:
        stats = train_with_args(config=None, args=args)
        logging.success(prettydict(stats, title="Overview of stats"))
        save_dict(args.output_dir + savename(args) + "_stats.json", stats)
        logging.info(f"Stats saved in {args.output_dir + savename(args)}_stats.json")


