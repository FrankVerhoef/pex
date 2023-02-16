"""
python run/eval.py
to run with defaults
"""
import torch
import torch.nn as nn
import random

from transformers import AutoTokenizer
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import BertClassifier
from dataset.msc_summary import MSC_Turns, extra_tokens
from dataset.vocab import Vocab, PAD_TOKEN, START_TOKEN, END_TOKEN


def eval(model, dataloader, vocab, decoder_max):

    model.eval()

    for batch in iter(dataloader):

        xs, ys, xs_len, ys_len = batch

        with torch.no_grad():
            output = model(xs, xs_len, max=decoder_max)
            pred = output.argmax(dim=-1)

        print_predictions(xs, ys, pred, vocab)


def eval_bert(model, dataloader, tokenizer):

    num_correct, num_eval = 0, 0
    model.eval()

    for batch in iter(dataloader):

        (input_ids, attention_mask, token_type_ids), y = batch

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            pred = logits.argmax(dim=1)
            
        num_correct += pred.eq(y).sum().item()
        num_eval += len(y)
        print_bert_predictions(input_ids, y, pred, tokenizer)

    return {"test_acc": num_correct/num_eval}


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
    parser.add_argument("--load", type=str, help="filename of model to load", required=True)
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
    parser.add_argument("--testdata", type=str, default="msc/msc_personasummary/session_1/train.txt", help="Dataset file for testing")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Max number of unique token (excluding special tokens)")
    parser.add_argument("--test_samples", type=int, default=10, help="Max number of test samples")
    
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.task == 'classify':

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        num_added_toks = tokenizer.add_tokens(extra_tokens)

        model = BertClassifier()
        model.bert.resize_token_embeddings(len(tokenizer))
        testdata = MSC_Turn_Facts(args.datadir + args.testdata, tokenizer, len_context=2, max_samples=args.test_samples)

    elif args.task == 'generate':

        vocab = Vocab()
        vocab.load("vocab_{}".format(args.vocab_size))
        testdata = MSC_Turns(args.datadir + args.testdata, vocab.text2vec, len_context=2, max_samples=args.test_samples)

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
    
    model.load_state_dict(torch.load(args.checkpoint_dir + args.load))

    test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=1, shuffle=True, collate_fn=testdata.batchify)

    if args.model == 'bert':
        eval_stats = eval_bert(model, test_loader, tokenizer)
        for k, v in eval_stats.items():
            print("{:<8}: {}".format(k, v))
    elif args.model == 'seq2seq':
        eval(model, test_loader, vocab, decoder_max=args.decoder_max)

