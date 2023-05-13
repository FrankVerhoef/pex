###
### Class to read the MSC summary dataset, and preprocess the data.
###


import torch

from torchmetrics import MeanMetric, TranslationEditRate
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.infolm import InfoLM
 
from torch.utils.data import Dataset

import json
import random
import itertools
from collections import Counter

import utils.logging as logging
from utils.general import padded_tensor_left
from utils.plotting import plot_heatmap

TER_MAXIMUM = 0.5
BERT_MINIMUM = 0.8

def calc_stats(predicted_summaries, target_summaries):

    def _format_scores(target_sentences, recall_correct):
        format_string = "{:20} " + "{:4} " * len(recall_correct)
        result = ""
        for s, r in zip(target_sentences, recall_correct):
            result += format_string.format(s, *r.int().tolist()) + '\n'
        return result

    ter_metric = TranslationEditRate(return_sentence_level_score=True)
    ter_per_summary = []
    ter_precisions = [] # MeanMetric()
    ter_recalls = [] # MeanMetric()

    bert_metric = BERTScore(model_name_or_path='microsoft/deberta-xlarge-mnli')
    bert_per_summary = []
    bert_precisions = []
    bert_recalls = []

    # infolm_metric = InfoLM(model_name_or_path='google/bert_uncased_L-2_H-128_A-2', return_sentence_level_score=True)
    # infolm_per_summary = []

    for prediction, target in zip(predicted_summaries, target_summaries):
        pred_sentences = [p.lower() for p in prediction.replace('. ', '\n').replace('.', '').split('\n') if p != '']
        target_sentences = [t.lower() for t in target.replace('. ', '\n').replace('.', '').split('\n') if t != '']
        combinations = list(itertools.product(pred_sentences, target_sentences))
        
        ter_scores = ter_metric(*zip(*combinations))
        ter_scores = ter_scores[1].view(-1,len(target_sentences))
        matching_predictions = ter_scores <= TER_MAXIMUM
        ter_precision = torch.any(matching_predictions, dim=1).float().mean().item()
        ter_recall = torch.any(matching_predictions, dim=0).float().mean().item()

        bert_scores = bert_metric(*zip(*combinations))
        bert_scores = torch.as_tensor(bert_scores['f1']).view(-1,len(target_sentences))
        matching_predictions = bert_scores >= BERT_MINIMUM
        bert_precision = torch.any(matching_predictions, dim=1).float().mean().item()
        bert_recall = torch.any(matching_predictions, dim=0).float().mean().item()

        # plot_heatmap(ter_scores.permute(1,0), target_sentences, pred_sentences)
        # plot_heatmap(bert_scores.permute(1,0), target_sentences, pred_sentences)

        ter_per_summary.append(2 * ter_precision * ter_recall / (ter_precision + ter_recall))
        ter_precisions.append(ter_precision)
        ter_recalls.append(ter_recall)

        bert_per_summary.append(2 * bert_precision * bert_recall / (bert_precision + bert_recall))
        bert_precisions.append(bert_precision)
        bert_recalls.append(bert_recall)

    stats = {
        "ter": ter_per_summary,
        "ter_precision": ter_precisions, #.compute(),
        "ter_recall": ter_recalls, #.compute(),
        "bert": bert_per_summary,
        "bert_precision": bert_precisions,
        "bert_recall": bert_recalls,
        # "infolm": infolm_per_summary,
    }
    return stats


class MSC_Summaries(Dataset):

    tokenizer = None
    speaker_prefixes = None
    nofact_token = None
        
    @classmethod
    def set(cls, tokenizer, speaker_prefixes, nofact_token):
        assert True if speaker_prefixes is None else len(speaker_prefixes) == 2, "If speaker_prefixes are set, 2 values are required"
        cls.tokenizer = tokenizer
        cls.speaker_prefixes = speaker_prefixes
        cls.nofact_token = nofact_token

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Turns')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")        
        group.add_argument("--sessions", default=[1], nargs='+', help="MSC sessions to include in dataset")
        return parser

    def __init__(self, basedir='./', sessions=[1], subset='train', max_samples=None):
        super(MSC_Summaries, self).__init__()
        self.sessions = sessions
        self.subset = subset
        dialogues = []
        for s in self.sessions:
            filepath = f"{basedir}session_{s}/{subset}.txt"
            try:
                with open(filepath, "r") as f:
                    for line in f:
                        dialogues.append(json.loads(line))
            except FileNotFoundError:
                logging.warning(f"File '{filepath}' not found -> skipped")
        self.turns, self.summaries = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        turns, summaries = [], []
        
        for d in dialogues:
            start_index = len(d["dialog"]) % 2
            utterances = []
            for i in range(start_index, len(d["dialog"]), 2):
                t = d["dialog"][i].get("text","")
                utterances.append((d["dialog"][i].get("text",""), d["dialog"][i+1].get("text","")))

            turns.append(utterances)
            summaries.append('\n'.join(d['dialog'][-1]['agg_persona_list']))
        
        if max_samples is not None:
            if max_samples < len(turns):
                indices = random.sample(range(len(turns)), max_samples)
                turns = [turns[i] for i in indices]
                summaries = [summaries[i] for i in indices]

        return turns, summaries
        
    def __len__(self):
        return len(self.summaries)
    
    def __getitem__(self, i):
        """
        Each item is a tuple with two elements:
        0) a list of turns (each turn has two utterances)
        1) a summary (which is a string of persona sentences joined by '\n')
        """

        if self.speaker_prefixes is not None:
            utterances = [
                self.speaker_prefixes[0] + ' ' + t[0] + ' ' + self.speaker_prefixes[1] + ' ' + t[1]
                for t in self.turns[i]
            ]
        else:
            utterances = [
                t[0] + ' ' + t[1]
                for t in self.turns[i]                
            ]

        return utterances, self.summaries[i]

    @classmethod
    def item_measurements(cls, item):
        stats = {
            "inputwords": len(' '.join(item[0]).split()), 
            "labelwords": len(item[1].split()), 
            "labelsentences": len(item[1].split('\n'))
        }       
        return stats

    def measurements(self):

        num_samples = self.__len__()
        allitem_measurements = [self.item_measurements(self.__getitem__(i)) for i in range(self.__len__())]
        inputwords_per_sample = Counter([m["inputwords"] for m in allitem_measurements])
        labelwords_per_sample = Counter([m["labelwords"] for m in allitem_measurements])
        totalwords_per_sample = Counter([m["inputwords"] + m["labelwords"] for m in allitem_measurements])
        labelsentences_per_sample = Counter([m["labelsentences"] for m in allitem_measurements])

        inputwords = sum([length * freq for length, freq in inputwords_per_sample.items()])
        labelwords = sum([length * freq for length, freq in labelwords_per_sample.items()])
        totalwords = sum([length * freq for length, freq in totalwords_per_sample.items()])
        labelsentences = sum([length * freq for length, freq in labelsentences_per_sample.items()])

        all_measurements = {
            "num_samples": num_samples,
            "inputwords": inputwords,
            "labelwords": labelwords,
            "totalwords": totalwords,
            "labelsentences": labelsentences,
            "avg_inputwords": inputwords / num_samples,
            "avg_labelwords": labelwords / num_samples,
            "avg_totalwords": totalwords / num_samples,
            "avg_labelsentences": labelsentences / num_samples,
            "inputwords_per_sample": sorted(inputwords_per_sample.items(), key=lambda x:x[0]),
            "labelwords_per_sample": sorted(labelwords_per_sample.items(), key=lambda x:x[0]),
            "totalwords_per_sample": sorted(totalwords_per_sample.items(), key=lambda x:x[0]),
            "labelsentences_per_sample": sorted(labelsentences_per_sample.items(), key=lambda x:x[0]),
        }

        return all_measurements


    @classmethod
    def batchify(cls, data):
        """
        Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert cls.tokenizer is not None, "Need to specify function to vectorize dataset"
        # assert self.tokenizer.padding_side == 'left', "Tokenizer padding_side must be 'left'"

        # seperate source and target sequences
        utterances, summaries = zip(*data)

        encoded_utterances = [
            cls.tokenizer(text=turns, padding=True, return_tensors="pt")
            for turns in utterances
        ]
        encoded_summaries = cls.tokenizer(text=summaries, padding=True, return_tensors='pt')

        return encoded_utterances, encoded_summaries

    def formatted_item(self, item):
        utterances, summary = item
        output = 'Utterances: ' + '\n\t' + '\n\t'.join(utterances) + '\n'
        output += 'Summary: ' + '\n\t' + summary.replace('\n', '\n\t')
        return output

    def evaluate(self, model, nofact_token='', device="cpu", decoder_max=20, print_max=20, log_interval=100):

        def print_predictions(text_in, text_out):

            x, y = text_in
            print('context:    ', x)
            print('target:     ', y)
            print('prediction: ', text_out)
            print('-' * 40)

        model = model.to(device)
        model.eval()
        target_summaries = []
        pred_summaries = []

        for i in range(self.__len__()):

            target_summary = self.__getitem__(i)[1]
            batch = self.batchify([self.__getitem__(i)])  # Batch with one sample
            encoded_utterances = batch[0][0]

            with torch.no_grad():
                pred_tokens = model.generate(
                    input_ids=encoded_utterances['input_ids'].to(device), 
                    attention_mask=encoded_utterances['attention_mask'].to(device),
                    min_length=2,
                    max_new_tokens=decoder_max, 
                    num_beams=5,
                    do_sample=True,
                )

            preds = self.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            pred_summary = '\n'.join([pred for pred in preds if pred != nofact_token])

            if print_max > 0:
                print(self.formatted_item(self.__getitem__(i)))
                print('Prediction: ' + '\n\t' + pred_summary.replace('\n', '\n\t') + '\n')
                print_max -= 1

            target_summaries.append(target_summary)
            pred_summaries.append(pred_summary)

            if (i + 1) % log_interval == 0:
                logging.verbose(f"Evaluated {i + 1}/{self.__len__()} samples")
        
        stats = calc_stats(pred_summaries, target_summaries)

        return stats

    @classmethod
    def predict(cls, data, model, nofact_token='', device="cpu", decoder_max=20):

        model = model.to(device)
        model.eval()
        pred_summaries = []

        for input_utterances in data:

            batch = cls.batchify([(input_utterances, "")])  # Batch with one sample, and empty summary
            encoded_utterances = batch[0][0]

            with torch.no_grad():
                pred_tokens = model.generate(
                    input_ids=encoded_utterances['input_ids'].to(device), 
                    attention_mask=encoded_utterances['attention_mask'].to(device),
                    min_length=2,
                    max_new_tokens=decoder_max, 
                    num_beams=5,
                    do_sample=True,
                )

            preds = cls.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            pred_summary = '\n'.join([pred for pred in preds if pred != nofact_token])

            pred_summaries.append(pred_summary)

        return pred_summaries


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from dataset.msc_summary_turns import MSC_Turns
    from models.bart_extractor import BartExtractor

    parser = argparse.ArgumentParser(description="Test MSC_Summary", conflict_handler="resolve")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--log_interval", type=int, default=10, help="report interval")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())    
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--load", type=str, default="", help="filename of model to load")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--datadir", type=str, default="./data/", help="Datadir")
    parser.add_argument("--basedir", type=str, default="msc/msc_personasummary/", help="Base directory for dataset")
    parser = MSC_Turns.add_cmdline_args(parser)
    parser = MSC_Summaries.add_cmdline_args(parser)
    parser = BartExtractor.add_cmdline_args(parser)
    parser.add_argument("--nofact_token", default='', type=str, help="Token to identify no_fact, default=''")
    args = parser.parse_args()

    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    # Settings for this test
    subset = 'test'
    test_samples = 5
    args.load = "trained_bart"
    args.speaker_prefixes = ["<other>", "<self>"]
    args.nofact_token = "<nofact>"
    args.add_tokens = ["<other>", "<self>", "<nofact>"]

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained(args.bart_base)
    if args.add_tokens is not None:
        num_added_toks = tokenizer.add_tokens(args.add_tokens)
    pad_token_id = tokenizer.pad_token_id
    nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
    assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

    MSC_Summaries.set(tokenizer=tokenizer, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
    train_set = MSC_Summaries(
        basedir=args.datadir + args.basedir, 
        sessions=args.sessions, 
        subset="train",         
    )
    m = MSC_Summaries.item_measurements(train_set[0])
    train_measurements = train_set.measurements()
    logging.report('\n'.join(["{}:\t{}".format(k, v) for k, v in train_measurements.items()]))

    msc_summaries = MSC_Summaries(
        basedir=args.datadir + args.basedir, 
        sessions=args.sessions, 
        subset=subset, 
        max_samples=test_samples, 
    )
    data = [msc_summaries[i] for i in range(test_samples)]

    for item in data:
        logging.verbose(msc_summaries.formatted_item(item))
        logging.verbose('-'*40)

    batch = msc_summaries.batchify(data)
    # logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)

    # Test the evaluation with BART model
    nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
    assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

    model = BartExtractor(bart_base=args.bart_base, nofact_token_id=nofact_token_id)
    model.bart.resize_token_embeddings(len(tokenizer))

    logging.info("Loading model from {}".format(args.checkpoint_dir + args.load))
    model.load_state_dict(torch.load(args.checkpoint_dir + args.load, map_location=torch.device(args.device)))

    pred_summaries = msc_summaries.predict([utterances for utterances, _ in data], model)
    logging.report(('\n----------------------------------------\n').join(pred_summaries))

    eval_kwargs = {'nofact_token': args.nofact_token, 'device': args.device, 'log_interval': args.log_interval, 'decoder_max': 30}
    eval_stats = msc_summaries.evaluate(model, **eval_kwargs)
    logging.report(eval_stats)