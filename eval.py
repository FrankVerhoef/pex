"""
python run/eval.py
to run with defaults
"""
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from torcheval.metrics.functional import binary_confusion_matrix, binary_accuracy, multiclass_accuracy, binary_f1_score, bleu_score

import transformers
from transformers import AutoTokenizer
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import BertClassifier, PrefixBert
from models.bart_extractor import BartExtractor, PrefixBart
from dataset.msc_summary_hf import MSC_Turns, PERSONA_TOKENS, NO_FACT_TOKEN
from dataset.vocab import Vocab, PAD_TOKEN, START_TOKEN, END_TOKEN

import utils.logging as logging

def eval(model, dataloader, vocab, decoder_max):

    model.eval()

    for batch in iter(dataloader):

        xs, ys, xs_len, ys_len = batch

        with torch.no_grad():
            output = model(xs, xs_len, max=decoder_max)
            pred = output.argmax(dim=-1)

        print_predictions(xs, ys, pred, vocab)


def eval_bart_text(model, dataset, tokenizer, decoder_max):

    model.eval()
    target_personas = []
    pred_personas = []
    target_facts = []
    pred_facts = []

    for i in range(len(dataset)):

        target_persona = dataset[i][1]
        batch = dataset.batchify([dataset[i]])

        with torch.no_grad():
            pred_tokens = model.generate(
                batch['input_ids'], 
                min_length=2,
                max_new_tokens=decoder_max, 
                num_beams=1,
                do_sample=False,
                # generation_config=model.gen_config
            )[0]
        pred_fact = pred_tokens[2] != model.nofact_token_id

        if pred_fact:
            pred_persona = tokenizer.decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            pred_persona = NO_FACT_TOKEN

        print_bart_predictions(dataset[i], pred_persona)

        if target_persona != NO_FACT_TOKEN:
            target_facts.append(1)
            target_personas.append(target_persona)
            pred_personas.append(pred_persona)
        else:
            target_facts.append(0)
        pred_facts.append(pred_fact)

    target_facts = torch.tensor(target_facts)
    pred_facts =  torch.tensor(pred_facts)
    
    try:
        bleu_4 = bleu_score(pred_personas, target_personas).item()
    except ValueError:
        bleu_4 = 0

    stats = {
        "test_acc": binary_accuracy(pred_facts, target_facts).item(),
        "f1": binary_f1_score(pred_facts, target_facts).item(),
        "cm": binary_confusion_matrix(pred_facts, target_facts).tolist(),
        "bleu": bleu_4
    }

    return stats

def eval_bart_data(model, dataloader, tokenizer):

    model.eval()

    for batch in iter(dataloader):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        y = batch['labels']

        with torch.no_grad():
            logprobs = model(input_ids, attention_mask, y)

        pred = logprobs.cpu().argmax(dim=-1)
        ignore_mask = batch['labels'].ne(model.bart.config.pad_token_id)
        correct = batch['labels'].eq(pred) * ignore_mask
        acc = (correct.sum() / ignore_mask.sum()).item() 

        print_bart_data(input_ids, y, pred, tokenizer)

        stats = {
            "acc": acc
        }

    return stats

def eval_bert(model, dataloader, tokenizer):

    all_labels = []
    all_preds = []
    model.eval()

    for batch in tqdm(iter(dataloader)):

        (input_ids, attention_mask, token_type_ids), y = batch

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            pred = logits.argmax(dim=1)
            
        all_labels.append(y)
        all_preds.append(pred)
        # print_bert_predictions(input_ids, y, pred, tokenizer)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    stats = {
        "test_acc": binary_accuracy(all_preds, all_labels).item(),
        "f1": binary_f1_score(all_preds, all_labels).item(),
        "cm": binary_confusion_matrix(all_preds, all_labels).tolist()
    }

    return stats

def print_bart_predictions(text_in, text_out):

    x, y = text_in
    print('-' * 40)
    print('context:    ', x)
    print('target:     ', y)
    print('prediction: ', text_out)

def print_bart_data(xs, ys, pred, tokenizer):

    for x, y, p in zip(xs, ys, pred):
        print('-' * 40)
        print('context:    ', tokenizer.decode(x))
        print('target:     ', tokenizer.decode(y))
        print('prediction: ', tokenizer.decode(p))

def print_bert_predictions(xs, ys, pred, tokenizer):

    for x, y, p in zip(xs, ys, pred):
        print('-' * 40)
        print('context:    ', tokenizer.decode(x))
        print('target:     ', y.item())
        print('prediction: ', p.item())

def print_predictions(xs, ys, pred, vocab):

    for x, y, p in zip(xs, ys, pred):
        try:
            p_len = list(p).index(vocab.tok2ind[END_TOKEN])
        except:
            p_len = len(p)
        print('-' * 40)
        print('context:    ', vocab.vec2text(x))
        print('target:     ', vocab.vec2text(y))
        print('prediction: ', vocab.vec2text(p[:p_len]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--load", type=str, default='', help="filename of model to load") #, required=True)
    parser.add_argument("--task", type=str, default="classify", choices=["generate", "classify"])
    parser.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")

    # Encoder and decoder model
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "bert", "bart", "prefixbart"], help="Encoder model")

    # Dataset
    parser.add_argument("--datadir", type=str, default="/Users/FrankVerhoef/Programming/PEX/data/", help="Datadir")
    parser.add_argument("--testdata", type=str, default="msc/msc_personasummary/session_1/train.txt", help="Dataset file for testing")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Max number of unique token (excluding special tokens)")
    parser.add_argument("--test_samples", type=int, default=10, help="Max number of test samples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

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
        "prefixbart": PrefixBart
    }[args.model].add_cmdline_args(parser)

    # Add cmdline arguments for task/dataset
    parser = {
        "classify": MSC_Turn_Facts,
        "generate": MSC_Turns
    }[args.task].add_cmdline_args(parser)

    args = parser.parse_args()

    if args.task == 'classify':

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if args.persona_identifier == "token":
            num_added_toks = tokenizer.add_tokens(PERSONA_TOKENS)
        
        model = PrefixBert('bert-base-uncased', prefix_size=args.prefix_size)
        model.bert.resize_token_embeddings(len(tokenizer))
        testdata = MSC_Turn_Facts(args.datadir + args.testdata, tokenizer, len_context=2, max_samples=args.test_samples)

    elif args.task == 'generate':

        if args.model == "seq2seq":
            vocab = Vocab()
            tokenizer = vocab.text2vec
            if args.persona_identifier == "token":
                vocab.add_special_tokens(PERSONA_TOKENS)
            traindata = MSC_Turns(args.datadir + args.traindata, tokenizer, len_context=2, persona_identifier=args.persona_identifier, max_samples=args.train_samples)
            vocab.add_to_vocab(traindata.corpus())
            if args.vocab_size is not None:
                vocab.cut_vocab(max_tokens=args.vocab_size)
            vocab.save("vocab_{}".format(len(vocab)))
            pad_token_id = vocab.tok2ind[PAD_TOKEN]
            start_token_id = vocab.tok2ind[START_TOKEN]
            vocab_size = len(vocab)
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
            model = PersonaExtractor(args.encoder, encoder_opts, args.decoder, decoder_opts, start_token=start_token_id)

        elif args.model[-4:] == "bart":
            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
            if args.persona_identifier == "token":
                tokenizer.add_special_tokens({'additional_special_tokens': PERSONA_TOKENS + [NO_FACT_TOKEN]})
            vocab_size = tokenizer.vocab_size
            pad_token_id = tokenizer.pad_token_id
            start_token_id = tokenizer.eos_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(NO_FACT_TOKEN)
            assert nofact_token_id != tokenizer.unk_token_id, "NO_FACT_TOKEN cannot be unknown token"
            bart_base = "facebook/bart-large-cnn" if args.load == "" else None
            if args.model == "bart":
                model = BartExtractor(bart_base=bart_base, nofact_token_id=nofact_token_id)
            else:
                model = PrefixBart(
                    bart_base=bart_base, 
                    nofact_token_id=nofact_token_id, 
                    freeze=args.freeze, 
                    enc_prefix_size=args.enc_prefix_size,
                    dec_prefix_size=args.dec_prefix_size,
                    prefix_aggr=args.prefix_aggr
                )
            model.bart.resize_token_embeddings(len(tokenizer))

    if args.load != '':
        logging.info("Loading model from {}".format(args.checkpoint_dir + args.load))
        model.load_state_dict(torch.load(args.checkpoint_dir + args.load))

    testdata = MSC_Turns(args.datadir + args.testdata, tokenizer, len_context=2, persona_identifier=args.persona_identifier, max_samples=args.test_samples)
    test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=True, collate_fn=testdata.batchify)

    if args.model == 'bert':
        eval_stats = eval_bert(model, test_loader, tokenizer)
        for k, v in eval_stats.items():
            print("{:<10}: {}".format(k, v))
    if args.model[-4:] == 'bart':
        if args.teacher_forcing:
            eval_stats = eval_bart_data(model, test_loader, tokenizer)
        else:
            eval_stats = eval_bart_text(model, testdata, tokenizer, decoder_max=args.decoder_max)
        report = '\n'.join(["{:<10}: {}".format(k, v) for k, v in eval_stats.items()])
        logging.report(report)
    elif args.model == 'seq2seq':
        eval(model, test_loader, vocab, decoder_max=args.decoder_max)

