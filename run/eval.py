"""
python run/eval.py
to run with defaults
"""
import torch
import random

from transformers import AutoTokenizer, PretrainedConfig
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import PrefixBert
from models.bart_extractor import BartExtractor, PrefixBart, BART_BASE
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder
from models.knowledge_grounded_generator.kg_utils import ConceptGraph
from dataset.msc_summary_turns import MSC_Turns
from dataset.tokenizer import Tokenizer, PAD_TOKEN, END_TOKEN, UNK_TOKEN
from dataset.msc_kg_sessions import KG_enriched_MSC_Session
from dataset.convai2 import ConvAI2

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
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert", "bart", "prefixbart", "kg_gen"], help="Encoder model")

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
    logging.info("Args: {}".format(args))

    # Add cmdline arguments for model
    parser = {
        "seq2seq": PersonaExtractor,
        "bert": PrefixBert,
        "bart": BartExtractor,
        "prefixbart": PrefixBart,
        "kg_gen": KnowledgeGroundedDecoder,
    }[args.model].add_cmdline_args(parser)

    # Add cmdline arguments for task/dataset
    parser = {
        "classify": MSC_Turn_Facts,
        "generate": MSC_Turns,
        "dialog": KG_enriched_MSC_Session,
    }[args.task].add_cmdline_args(parser)
    
    args = parser.parse_known_args()[0]
    if 1 in args.sessions:
        parser = ConvAI2.add_cmdline_args(parser)
    
    if args.model == "seq2seq":
        parser.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")

    args = parser.parse_args()

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
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'sessions': args.sessions,
            'tokenizer': tokenizer,
            'len_context': args.len_context,
            'speaker_prefixes': args.speaker_prefixes,
            'nofact_token': args.nofact_token,
            'batch_format': 'huggingface',
            'batch_pad_id': tokenizer.pad_token_id
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
            batch_format = "padded_sequences"

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
            tokenizer = AutoTokenizer.from_pretrained(BART_BASE)
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            pad_token_id = tokenizer.pad_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
            assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)
            batch_format = "huggingface"
            
            # Define model
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

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'sessions': args.sessions,
            'tokenizer': tokenizer,
            'len_context': args.len_context,
            'speaker_prefixes': args.speaker_prefixes,
            'nofact_token': args.nofact_token,
            'batch_format': batch_format,
            'batch_pad_id': pad_token_id
        } 
        testdata = MSC_Turns(subset='test', max_samples=args.test_samples, **dataset_config)

    elif args.task == "dialog":

        if args.model == "kg_gen":
        
            tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            model = KnowledgeGroundedDecoder(vars(args), tokenizer, config=PretrainedConfig())

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)
        
        kg = ConceptGraph(args.kg_datadir, args.kg)
        kg.build_reduced_graph(args.kg_datadir + args.dataset_concepts)
        if 1 in args.sessions:
            args.sessions = [(item if item != 1 else '-'.join(['1'] + args.convai2_version)) for item in args.sessions]
        dataset_config = {
            'basedir': args.datadir + args.basedir,
            'tokenizer': tokenizer,
            'batch_format': "huggingface",
            'batch_pad_id': tokenizer.pad_token_id
        } 
        testdata = KG_enriched_MSC_Session(vars(args), subset='test', kg=kg, max_samples=args.test_samples, **dataset_config)

    if args.load != '':
        logging.info("Loading model from {}".format(args.checkpoint_dir + args.load))
        model.load_state_dict(torch.load(args.checkpoint_dir + args.load))

    eval_kwargs = {'device': args.device, 'log_interval': args.log_interval}
    if args.task in ["generate", "dialog"]:
        if args.device == 'mps':
            eval_kwargs['device'] = 'cpu'
            logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
        eval_kwargs.update({'decoder_max': args.decoder_max})
    if args.task == "dialog":
        eval_kwargs.update({'batch_size': args.batch_size})

    logging.info("Evaluating model on {} samples of testdata in {} with arguments {}".format(len(testdata), args.basedir, eval_kwargs))
    eval_stats = testdata.evaluate(model, **eval_kwargs)
    report = '\n'.join(["{:<10}: {}".format(k, v) for k, v in eval_stats.items()])
    logging.report(report)

