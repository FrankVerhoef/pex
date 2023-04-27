"""
python run/eval.py
to run with defaults
"""
import torch
import torch.nn as nn
import random
from functools import partial
import json

from transformers import AutoTokenizer, PretrainedConfig
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import PrefixBert
from models.bart_extractor import BartExtractor, PrefixBart
from models.dialogpt import DialoGPT
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss
from models.knowledge_grounded_generator.kg_utils import ConceptGraph
from dataset.msc_summary_turns import MSC_Turns
from dataset.tokenizer import Tokenizer, PAD_TOKEN, END_TOKEN, UNK_TOKEN
from dataset.msc_kg_sessions import KG_enriched_MSC_Session
from dataset.msc_sessions import MSC_Session
from dataset.convai2 import ConvAI2

from run.main import evaluate
from utils.general import loadname_prefix, savename
import utils.logging as logging

     
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model", conflict_handler="resolve")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--load", type=str, default='', help="filename of model to load") #, required=True)
    parser.add_argument("--task", type=str, default="classify", choices=["generate", "classify", "dialog"])
    parser.add_argument("--log_interval", type=int, default=100, help="report interval")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])

    # Encoder and decoder model
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert", "bart", "prefixbart", "kg_gen", "dialogpt"], help="Encoder model")

    # Dataset
    parser.add_argument("--datadir", type=str, default="/Users/FrankVerhoef/Programming/PEX/data/", help="Datadir")
    parser.add_argument("--basedir", type=str, default="msc/msc_personasummary/", help="Base directory for dataset")
    parser.add_argument("--testdata", type=str, default="msc/msc_dialogue/session_2/test.txt", help="Dataset file for testing")

    parser.add_argument("--test_samples", type=int, default=None, help="Max number of test samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--print_max", type=int, default=20, help="Max number of test examples to print")

    args = parser.parse_known_args()[0]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare logging
    logging.set_log_level(args.loglevel)
    if args.logdir is not None:
        logging.add_file_handler(logdir=args.logdir)

    # Add cmdline arguments for model
    parser = {
        "seq2seq": PersonaExtractor,
        "bert": PrefixBert,
        "bart": BartExtractor,
        "prefixbart": PrefixBart,
        "kg_gen": KnowledgeGroundedDecoder,
        "dialogpt": DialoGPT,
    }[args.model].add_cmdline_args(parser)

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
    
    if args.model == "seq2seq":
        parser.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")

    args = parser.parse_args()
    logging.info("Args: {}".format('\n'.join(["{:20s}: {}".format(k, v) for k, v in vars(args).items()])))

    if args.task == 'classify':

        # Classify whether dialog turns contain a fact

        if args.model == "bert":

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            model = PrefixBert('bert-base-uncased', prefix_size=args.prefix_size, prefix_aggr=args.prefix_aggr)
            model.bert.resize_token_embeddings(len(tokenizer))

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        MSC_Turn_Facts.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'session': args.session
        } 
        testdata = MSC_Turn_Facts(subset='test', max_samples=args.test_samples, **dataset_config)

    elif args.task == 'generate':

        if args.model == "seq2seq":

            # Get tokenizer and dataset parameters: note this must be built with same parameters as during training
            args.save=loadname_prefix(args.load) # Necessary to retrieve the correct name for tokenizer
            tokenizer = Tokenizer.from_file(args.checkpoint_dir + savename(args) + '_tokenizer.json')
            pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
            eos_token_id = tokenizer.token_to_id(END_TOKEN)
            unk_token_id = tokenizer.token_to_id(UNK_TOKEN)
            nofact_token_id = tokenizer.token_to_id(args.nofact_token) if args.nofact_token != '' else eos_token_id
            assert nofact_token_id != unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)
            vocab_size = tokenizer.get_vocab_size()

            # Define model
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

        elif args.model[-4:] == "bart":

            # Get tokenizer and dataset parameters: note this must be built with same parameters as during training
            tokenizer = AutoTokenizer.from_pretrained(args.bart_base)
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            pad_token_id = tokenizer.pad_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
            assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

            # Define model
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

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        MSC_Turns.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'sessions': args.sessions
        } 
        testdata = MSC_Turns(subset='test', max_samples=args.test_samples, **dataset_config)
        collate_fn = partial(MSC_Turns.batchify, batch_format=model.batch_format, batch_pad_id=pad_token_id)

    elif args.task == "dialog":

        if args.model == "kg_gen":
        
            tokenizer = AutoTokenizer.from_pretrained(args.lm)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if args.add_tokens is not None:
                tokenizer.add_tokens(args.add_tokens)
            tokenizer.bos_token_id = tokenizer.eos_token_id
            model = KnowledgeGroundedDecoder(vars(args), tokenizer, config=PretrainedConfig())
            model.gpt2model.resize_token_embeddings(len(tokenizer))

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

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        if args.session == 1:
            args.session = '-'.join(['1'] + args.convai2_version)

        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'session': args.session,
            'include_persona': args.include_persona
        }

        if args.model == "kg_gen":
            KG_enriched_MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=args.speaker_prefixes)
            dataset_config.update({
                'num_hops': args.num_hops,
                'max_branch': args.max_branch,
                'max_concepts': args.max_concepts,
                'max_triples': args.max_triples,
                'overlapping_concepts': args.overlapping_concepts
            })
            testdata = KG_enriched_MSC_Session(subset='test', kg=kg, max_samples=args.test_samples, **dataset_config)

        elif args.model == "dialogpt":
            
            MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=args.speaker_prefixes)
            
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

            testdata = MSC_Session(subset='test', max_samples=args.test_samples, **dataset_config)
 
    if args.load != '':
        logging.info("Loading model from {}".format(args.checkpoint_dir + args.load))
        model.load_state_dict(torch.load(args.checkpoint_dir + args.load, map_location=torch.device('cpu')))

    logging.info("Start evaluation")
    eval_stats = evaluate(model, testdata, args)


