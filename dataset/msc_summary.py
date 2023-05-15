###
### Class to read the MSC summary dataset, and preprocess the data.
###


import torch

from torchmetrics import TranslationEditRate
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.infolm import InfoLM
from metrics.terp import TerpMetric, TERP_DIR, JAVA_HOME
 
from torch.utils.data import Dataset

import json
import random
import itertools
from collections import Counter

import utils.logging as logging
from utils.plotting import plot_heatmap

TER_MAXIMUM = 0.6
BERT_MINIMUM = 0.75
TERP_MAXIMUM = 0.6

def calc_stats(predicted_summaries, target_summaries, indices, metrics=None, tmpdir='./'):

    if metrics is None:
        metrics = metric.keys()

    metric = {
        "ter": TranslationEditRate(return_sentence_level_score=True),
        "bert": BERTScore(model_name_or_path='microsoft/deberta-xlarge-mnli'),
        "terp": TerpMetric(terp_dir=TERP_DIR, java_home=JAVA_HOME, tmp_dir=tmpdir),
        # infolm_metric = InfoLM(model_name_or_path='google/bert_uncased_L-2_H-128_A-2', return_sentence_level_score=True)
    }
    stats_dict = {}
    for m in metrics:
        stats_dict[m] = {"f1s": [], "precisions": [], "recalls": []} 

    result_dict = {}

    for dialog_nr, prediction, target in zip(indices, predicted_summaries, target_summaries):
        pred_sentences = [p.lower() for p in prediction.replace('. ', '\n').replace('.', '').split('\n') if p != '']
        target_sentences = [t.lower() for t in target.replace('. ', '\n').replace('.', '').split('\n') if t != '']
        combinations = list(itertools.product(pred_sentences, target_sentences))
        result_dict[dialog_nr] = {
            "pred_sentences": pred_sentences,
            "target_sentences": target_sentences,
            "num_combinations": len(combinations)
        }
        if "ter" in metrics:
            metric["ter"].update(*zip(*combinations))
        if "bert" in metrics:
            metric["bert"].update(*zip(*combinations))
        if "terp" in metrics:
            metric["terp"].update(dialog_nr, *zip(*combinations))

    all_scores = {m: metric[m].compute() for m in metrics}
    
    i_start = 0
    for dialog_nr in result_dict.keys():
        r = result_dict[dialog_nr]
        i_end = i_start + r["num_combinations"]

        if "ter" in metrics:
            scores = all_scores["ter"][1][i_start:i_end]
            scores = scores.view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)
            matching_predictions = scores <= TER_MAXIMUM
            precision = torch.any(matching_predictions, dim=0).float().mean().item()
            recall = torch.any(matching_predictions, dim=1).float().mean().item()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            r["ter"] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall}

            stats_dict["ter"]["f1s"].append(f1)
            stats_dict["ter"]["precisions"].append(precision)
            stats_dict["ter"]["recalls"].append(recall)
        
        if "bert" in metrics:
            scores = torch.as_tensor(all_scores["bert"]['f1'][i_start:i_end])
            scores = scores.view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)
            matching_predictions = scores >= BERT_MINIMUM
            precision = torch.any(matching_predictions, dim=0).float().mean().item()
            recall = torch.any(matching_predictions, dim=1).float().mean().item()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            r["bert"] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall}

            stats_dict["bert"]["f1s"].append(f1)
            stats_dict["bert"]["precisions"].append(precision)
            stats_dict["bert"]["recalls"].append(recall)

        if "terp" in metrics:
            scores = all_scores["terp"][dialog_nr]
            scores = scores.view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)
            matching_predictions = scores <= TERP_MAXIMUM
            precision = torch.any(matching_predictions, dim=0).float().mean().item()
            recall = torch.any(matching_predictions, dim=1).float().mean().item()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            r["terp"] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall}

            stats_dict["terp"]["f1s"].append(f1)
            stats_dict["terp"]["precisions"].append(precision)
            stats_dict["terp"]["recalls"].append(recall)

        i_start = i_end

    return stats_dict, result_dict

def plot_heatmaps(results_dict, session, subset, savedir):
    
    for dialog_nr in results_dict.keys():
        r = results_dict[dialog_nr]

        if "ter" in r.keys():
            plot_heatmap(
                scores=r["ter"]["scores"], 
                threshold=TER_MAXIMUM,
                criterion = lambda x, threshold: x <= threshold,
                targets=r["target_sentences"], 
                predictions=r["pred_sentences"], 
                title=f"TER heatmap MSC_Summary session_{session}/{subset}, dialog {dialog_nr}\n(threshold={TER_MAXIMUM:.2f})"
            ).figure.savefig(f"{savedir}ter_heatmap_session_{session}_{subset}_{dialog_nr:06d}.jpg")

        if "bert" in r.keys():
            plot_heatmap(
                scores=r["bert"]["scores"], 
                threshold=BERT_MINIMUM,
                criterion = lambda x, threshold: x >= threshold,
                targets=r["target_sentences"], 
                predictions=r["pred_sentences"], 
                title=f"BERT heatmap MSC_Summary session_{session}/{subset}, dialog {dialog_nr}\n(threshold={BERT_MINIMUM:.2f})"
            ).figure.savefig(f"{savedir}bert_heatmap_session_{session}_{subset}_{dialog_nr:06d}.jpg")

        if "terp" in r.keys():
            plot_heatmap(
                scores=r["terp"]["scores"], 
                threshold=TERP_MAXIMUM,
                criterion = lambda x, threshold: x <= threshold,
                targets=r["target_sentences"], 
                predictions=r["pred_sentences"], 
                title=f"TERp heatmap MSC_Summary session_{session}/{subset}, dialog {dialog_nr}\n(threshold={TERP_MAXIMUM:.2f})"
            ).figure.savefig(f"{savedir}terp_heatmap_session_{session}_{subset}_{dialog_nr:06d}.jpg")

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
        group = parser.add_argument_group('MSC_Summary')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")
        group.add_argument("--nofact_token", default='', type=str, help="Token to identify no_fact, default=''")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")   
        group.add_argument("--session", default=1, type=int, help="MSC session to include in dataset")
        return parser

    def __init__(self, basedir='./', session=1, subset='train', max_samples=None):
        super(MSC_Summaries, self).__init__()
        self.session = session
        self.subset = subset
        dialogues = []
        filepath = f"{basedir}session_{session}/{subset}.txt"
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
        
        self.indices = list(range(len(dialogues)))
        if max_samples is not None:
            if max_samples < len(turns):
                indices = random.sample(range(len(turns)), max_samples)
                turns = [turns[i] for i in indices]
                summaries = [summaries[i] for i in indices]
                self.indices = indices

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
        allitem_measurements = [self.item_measurements(self[i]) for i in range(len(self))]
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

    def evaluate(self, model, metrics=None, nofact_token='', device="cpu", decoder_max=20, print_max=20, log_interval=100, tmpdir='./'):

        model = model.to(device)
        model.eval()
        target_summaries = []
        pred_summaries = []

        logging.info(f"Start evaluation of model {model.__class__.__name__} on metrics: {','.join(metrics) if metrics is not None else 'all'}")
        for i in range(len(self)):

            target_summary = self[i][1]
            batch = self.batchify([self[i]])  # Batch with one sample
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
                print(self.formatted_item(self[i]))
                print('Prediction: ' + '\n\t' + pred_summary.replace('\n', '\n\t') + '\n')
                print_max -= 1

            target_summaries.append(target_summary)
            pred_summaries.append(pred_summary)

            if (i + 1) % log_interval == 0:
                logging.verbose(f"Evaluated {i + 1}/{len(self)} samples")
        
        stats, results_dict = calc_stats(pred_summaries, target_summaries, self.indices, metrics=metrics, tmpdir=tmpdir)

        return stats, results_dict

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

    parser = MSC_Summaries.add_cmdline_args(parser)
    parser = BartExtractor.add_cmdline_args(parser)
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare logging
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    # Settings for this test
    subset = 'test'
    test_samples = 10
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
        session=args.session, 
        subset="train",         
    )
    m = MSC_Summaries.item_measurements(train_set[0])
    train_measurements = train_set.measurements()
    logging.report('Measurements for session_{args.session}/train\n' + '\n'.join(["{}:\t{}".format(k, v) for k, v in train_measurements.items()]))

    msc_summaries = MSC_Summaries(
        basedir=args.datadir + args.basedir, 
        session=args.session, 
        subset=subset, 
        max_samples=test_samples, 
    )
    data = [msc_summaries[i] for i in range(test_samples)]

    for item in data:
        logging.verbose(msc_summaries.formatted_item(item))
        logging.verbose('-'*40)

    batch = msc_summaries.batchify(data)
    logging.spam(batch)

    # Prepare BART model for prediction and evaluation
    nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
    assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

    model = BartExtractor(bart_base=args.bart_base, nofact_token_id=nofact_token_id)
    model.bart.resize_token_embeddings(len(tokenizer))

    logging.info("Loading model {} from {}".format(model.__class__.__name__, args.checkpoint_dir + args.load))
    model.load_state_dict(torch.load(args.checkpoint_dir + args.load, map_location=torch.device(args.device)))

    # Test predictions
    pred_summaries = msc_summaries.predict([utterances for utterances, _ in data], model)
    logging.report(('\n----------------------------------------\n').join(pred_summaries))

    # Run evaluation
    eval_kwargs = {
        'metrics': ["terp", "ter", "bert"], 
        'nofact_token': args.nofact_token, 
        'device': args.device, 
        'log_interval': args.log_interval, 
        'decoder_max': 30,
        'tmpdir': '/Users/FrankVerhoef/Programming/PEX/output/'
    }
    eval_stats, results_dict = msc_summaries.evaluate(model, **eval_kwargs)
    logging.info(eval_stats)
    logging.report('\n'.join([
        f"{metric}_{k}:\t{sum(v)/len(v):.4f}"
        for metric, stats in eval_stats.items() for k, v in stats.items()
        if isinstance(v, list)
    ]))

    # Save results
    with open(f"./output/MSC_Summary_session_{msc_summaries.session}_{msc_summaries.subset}_evalresults.json", "w") as f:
        f.write(json.dumps(results_dict, sort_keys=True, indent=2))
    plot_heatmaps(results_dict, msc_summaries.session, msc_summaries.subset, savedir='./output/')