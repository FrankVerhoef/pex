###
### Class to read the MSC summary dataset, and preprocess the data.
###


import torch

from torchmetrics import TranslationEditRate
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.infolm import InfoLM
from torchmetrics.functional.text.rouge import rouge_score
from metrics.terp import TerpMetric, TERP_DIR, JAVA_HOME
from metrics.nli import NLIMetric
 
from torch.utils.data import Dataset

import json
import random
import itertools
from functools import partial
from collections import Counter
import textwrap

from utils.general import prettydict
import utils.logging as logging
from utils.plotting import plot_heatmap
from utils.plotting import save_dialogue_fig


##
## Functions to calculate evaluation statistics
##

TER_MAXIMUM = 0.75
BERT_MINIMUM = 0.75
TERP_MAXIMUM = 0.75
NLI_MINIMUM = 0.5


class ROUGE_List:

    rouge = partial(rouge_score, tokenizer=lambda s: s.replace('\n', ' ').split(' '), rouge_keys=('rougeL'))

    def __init__(self):
        self.ids = []
        self.preds = []
        self.targets = []


    def update(self, id, prediction, target):
        self.ids.append(id)
        self.preds.append(prediction)
        self.targets.append(target)

    def compute(self):
        scores = [
            self.rouge(p, t) for id, p, t in zip(self.ids, self.preds, self.targets)
        ]
        return scores

def calc_stats(predicted_summaries, target_summaries, indices, metrics=None):

    metric = {
        "ter": TranslationEditRate(return_sentence_level_score=True),
        "bert": BERTScore(model_name_or_path='microsoft/deberta-xlarge-mnli'),
        "terp": TerpMetric(),
        "nli": NLIMetric(),
        "rougeL": ROUGE_List(),
    }

    if metrics is None:
        metrics = list(metric.keys())

    assert len(set(metrics).difference(metric.keys())) == 0, f"Unknown metric in metrics: {metrics}; choose from {list(metric.keys())}"
    metric_keys = list(set(metrics))
    if "rougeL" in metrics:
        metric_keys.remove("rougeL")
        rouge_submetrics = ['rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall']
    if "nli" in metrics:
        metric_keys.remove("nli")
        metric_keys += ["nli" + suffix for suffix in ["_tp", "_pt", "_avg"]] if "nli" in metrics else []
    sub_metrics = ['f1', 'precision', 'recall', 'prec_strength', 'rec_strength']

    result_dict = {}
    for index, prediction, target in zip(indices, predicted_summaries, target_summaries):
        dialog_id = index["dialog_id"]
        pred_sentences = [p.lower() for p in prediction.replace('. ', '\n').replace('.', '').split('\n') if p != '']
        target_sentences = [t.lower() for t in target.replace('. ', '\n').replace('.', '').split('\n') if t != '']
        if len(pred_sentences) == 0:
            pred_sentences = [' ']
        if len(target_sentences) == 0:
            target_sentences = [' ']
        combinations = list(itertools.product(pred_sentences, target_sentences))
        result_dict[dialog_id] = {
            "convai_id": index["convai_id"],
            "pred_sentences": pred_sentences,
            "target_sentences": target_sentences,
            "num_combinations": len(combinations)
        }
        logging.debug(f"calc_stats {dialog_id}: update {len(combinations)} combinations")
        if "ter" in metrics:
            metric["ter"].update(*zip(*combinations))
        if "bert" in metrics:
            metric["bert"].update(*zip(*combinations))
        if "terp" in metrics:
            metric["terp"].update(dialog_id, *zip(*combinations))
        if "nli" in metrics:
            for comb_id, (pred, target) in enumerate(combinations):
                metric["nli"].update((dialog_id, "pt", comb_id), pred, target) # entailment score from prediction to target
                metric["nli"].update((dialog_id, "tp", comb_id), target, pred) # entailment score from target to prediction
        if "rougeL" in metrics:
            metric["rougeL"].update(dialog_id, prediction, target)
            # result_dict[dialog_id]["rougeL"] = rouge_score(prediction, target)

    # Compute all the metrics
    all_scores = {m: metric[m].compute() for m in metrics}

    # Transform the resulting NLI-metrics to two lists: 'tp' is from target to prediction; 'pt' is from prediction to target
    if "nli" in metrics:
        nli_scores = {}
        for (dialog_id, direction, _), score in all_scores["nli"].items():
            if dialog_id in nli_scores.keys():
                if direction in nli_scores[dialog_id].keys():
                    nli_scores[dialog_id][direction].append(score)
                else:
                    nli_scores[dialog_id].update({direction: [score]})
            else:
                nli_scores[dialog_id] = {direction: [score]}
        all_scores["nli"] = nli_scores

    # Calculate the submetrics (precision, recall and F1 scores etc) for each of the metrics
    i_start = 0
    for i, dialog_id in enumerate(result_dict.keys()):
        r = result_dict[dialog_id]
        logging.debug(f"calc_stats {dialog_id}: compute {metrics}")
        i_end = i_start + r["num_combinations"]

        if "ter" in metrics:
            scores = all_scores["ter"][1][i_start:i_end]
            scores = scores.view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)

            # precision, recall, F1
            matching_predictions = scores <= TER_MAXIMUM
            precision = torch.any(matching_predictions, dim=0).float().mean().item()
            recall = torch.any(matching_predictions, dim=1).float().mean().item()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            # strength
            prec_strength = (1 - scores).max(dim=0).values.mean().item()
            rec_strength = (1 - scores).max(dim=1).values.mean().item()

            r["ter"] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall, "prec_strength": prec_strength, "rec_strength": rec_strength}
        
        if "bert" in metrics:
            scores = torch.as_tensor(all_scores["bert"]['f1'][i_start:i_end])
            scores = scores.view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)

            # precision, recall, F1
            matching_predictions = scores >= BERT_MINIMUM
            precision = torch.any(matching_predictions, dim=0).float().mean().item()
            recall = torch.any(matching_predictions, dim=1).float().mean().item()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            # strength
            prec_strength = scores.max(dim=0).values.mean().item()
            rec_strength = scores.max(dim=1).values.mean().item()

            r["bert"] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall, "prec_strength": prec_strength, "rec_strength": rec_strength}

        if "terp" in metrics:
            scores = all_scores["terp"][dialog_id]
            scores = scores.view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)

            # precision, recall, F1
            matching_predictions = scores <= TERP_MAXIMUM
            precision = torch.any(matching_predictions, dim=0).float().mean().item()
            recall = torch.any(matching_predictions, dim=1).float().mean().item()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            # strength
            prec_strength = (1 - scores).max(dim=0).values.mean().item()
            rec_strength = (1 - scores).max(dim=1).values.mean().item()

            r["terp"] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall, "prec_strength": prec_strength, "rec_strength": rec_strength}

        if "nli" in metrics:
            nli_scores = {}
            nli_scores["tp"] = torch.tensor(all_scores["nli"][dialog_id]["tp"])
            nli_scores["pt"] = torch.tensor(all_scores["nli"][dialog_id]["pt"])
            nli_scores["avg"] = (nli_scores["tp"] + nli_scores["pt"]) / 2
            r["nli"] = {}
            for key in ["tp", "pt", "avg"]:
                scores = nli_scores[key].view(len(r["pred_sentences"]), len(r["target_sentences"])).permute(1,0)

                # precision, recall, F1
                matching_predictions = scores >= NLI_MINIMUM
                precision = torch.any(matching_predictions, dim=0).float().mean().item()
                recall = torch.any(matching_predictions, dim=1).float().mean().item()
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

                # strength
                prec_strength = scores.max(dim=0).values.mean().item()
                rec_strength = scores.max(dim=1).values.mean().item()

                r["nli_" + key] = {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall, "prec_strength": prec_strength, "rec_strength": rec_strength}

        # Calculate length difference
        r["numwords_factor"] = len(' '.join(r['pred_sentences'])) / max(len(' '.join(r['target_sentences'])), 1)
        r["numfacts_factor"] = len(r['pred_sentences']) / max(len(r['target_sentences']), 1)

        # Get ROUGE scores from all_scores
        if 'rougeL' in metrics:
            r["rougeL"] = {k: v.item() for k, v in all_scores["rougeL"][i].items()}

        i_start = i_end

    # Summarize all metrics in stats_dict
    stats_dict = {}
    num_dialogues = len(result_dict.keys())
    for metric in metric_keys:
        for sub_metric in sub_metrics:
            values = [result_dict[dialog_id][metric][sub_metric] for dialog_id in result_dict.keys()]
            stats_dict[f"{metric}_{sub_metric}"] = sum(values) / num_dialogues
    stats_dict["numwords_factor"] = sum([result_dict[dialog_id]["numwords_factor"] for dialog_id in result_dict.keys()]) / num_dialogues
    stats_dict["numfacts_factor"] = sum([result_dict[dialog_id]["numfacts_factor"] for dialog_id in result_dict.keys()]) / num_dialogues
    if "rougeL" in metrics:
        for dialog_id in result_dict.keys():
            for sub_metric in rouge_submetrics:
                stats_dict[sub_metric] = sum([result_dict[dialog_id]["rougeL"][sub_metric] for dialog_id in result_dict.keys()]) / num_dialogues

    return stats_dict, result_dict

def plot_heatmaps(results_dict, session, subset, savedir):
    
    for dialog_id in results_dict.keys():
        r = results_dict[dialog_id]

        if "ter" in r.keys():
            plot_heatmap(
                scores=r["ter"]["scores"], 
                threshold=TER_MAXIMUM,
                criterion = lambda x, threshold: x <= threshold,
                targets=r["target_sentences"], 
                predictions=r["pred_sentences"], 
                title="" #f"TER heatmap MSC_Summary session_{session}/{subset}, dialog {r['convai_id']}\n(threshold={TER_MAXIMUM:.2f})"
            ).figure.savefig(f"{savedir}ter_heatmap_session_{session}_{subset}_{r['convai_id']}.pdf", dpi=300, format='pdf', bbox_inches='tight')

        if "bert" in r.keys():
            plot_heatmap(
                scores=r["bert"]["scores"], 
                threshold=BERT_MINIMUM,
                criterion = lambda x, threshold: x >= threshold,
                targets=r["target_sentences"], 
                predictions=r["pred_sentences"], 
                title="" #f"BERT heatmap MSC_Summary session_{session}/{subset}, dialog {r['convai_id']}\n(threshold={BERT_MINIMUM:.2f})"
            ).figure.savefig(f"{savedir}bert_heatmap_session_{session}_{subset}_{r['convai_id']}.pdf", dpi=300, format='pdf', bbox_inches='tight')

        if "terp" in r.keys():
            plot_heatmap(
                scores=r["terp"]["scores"], 
                threshold=TERP_MAXIMUM,
                criterion = lambda x, threshold: x <= threshold,
                targets=r["target_sentences"], 
                predictions=r["pred_sentences"], 
                title="" #f"TERp heatmap MSC_Summary session_{session}/{subset}, dialog {r['convai_id']}\n(threshold={TERP_MAXIMUM:.2f})"
            ).figure.savefig(f"{savedir}terp_heatmap_session_{session}_{subset}_{r['convai_id']}.pdf", dpi=300, format='pdf', bbox_inches='tight')

        if len([m for m in r.keys() if m[:4] == "nli_"]):
            for key in ["tp", "pt", "avg"]:
                plot_heatmap(
                    scores=r["nli_" + key]["scores"], 
                    threshold=NLI_MINIMUM,
                    criterion = lambda x, threshold: x >= threshold,
                    targets=r["target_sentences"], 
                    predictions=r["pred_sentences"], 
                    title="" #f"NLI_{key} heatmap MSC_Summary session_{session}/{subset}, dialog {r['convai_id']}\n(threshold={NLI_MINIMUM:.2f})"
                ).figure.savefig(f"{savedir}nli_{key}_heatmap_session_{session}_{subset}_{r['convai_id']}.pdf", dpi=300, format='pdf', bbox_inches='tight')

##
## Definition of the dataset with Multi-Session Chat summaries
##

class MSC_Summaries(Dataset):

    tokenizer = None
    len_context = 2
    speaker_prefixes = None
    nofact_token = None
    OTHER = 0
    SELF = 1
        
    @classmethod
    def set(cls, tokenizer=None, len_context=2, speaker_prefixes=None, nofact_token=''):
        assert True if speaker_prefixes is None else len(speaker_prefixes) == 2, "If speaker_prefixes are set, 2 values are required"
        assert len_context == 2, f"Invalid setting for len_context '{len_context}'; currently only works with len_context=2"
        cls.tokenizer = tokenizer
        cls.len_context = len_context
        cls.speaker_prefixes = speaker_prefixes
        cls.nofact_token = nofact_token

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Summary')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")
        group.add_argument("--nofact_token", default='', type=str, help="Token to identify no_fact, default=''")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        group.add_argument("--len_context", default=2, type=int, help="Number of utterances to include in context")
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
        self.indices, self.turns, self.dialogues, self.summaries = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        """
        Format of a dialogue: Dict
            - "dialog": List with Dicts, each Dict is an utterance
                - "id": String
                - "text": String
                - "convai_id": String
                - "persona_text": String
                - "problem_data": Dict; 
                    NOTE: In all utterances, except the last in the list, this contains element "persona". In last utterance, the dict contains:
                    - "persona": String
                    - "prompt_time": String with indication of duration
                    - "task_duration": Float
                    - "followup": String
                    - "newfacts": String
                    - "task_time": String with time
                    - "hit_id": String with alphanumeric characters
                    - "worker_id": String with alphanumeric characters
                    - "initial_data_id": String with id
                - "agg_persona_list": List with Strings
            - "followup": String (Sentence)
            - "newfact": String (Sentence)
            - "initial_data_id": String (id) ==> refers to ID in the original ConvAI2 dataset
            - "init_personachat": Dict with the initial persona sentences
                - "init_personas": List with two lists with Strings
                    - 0: persona sentences Speaker 1
                    - 1: persona sentances Speaker 2
        """
        turns, summaries, dialogs, ids = [], [], [], []

        # If max_samples is set, and lower than total number of summaries, then take a random sample
        selection = list(range(len(dialogues)))
        if max_samples is not None:
            if max_samples < len(dialogues):
                selection = random.sample(range(len(dialogues)), max_samples)

        for dialog_id in selection:
            d = dialogues[dialog_id]

            # Start with utterance that is a multiple of len_context from the last utterance
            start_index = (len(d["dialog"]) - self.len_context) % 2
            utterances = []

            # Collect turns that end with the 'other' speaker (so step size is 2)
            for i in range(start_index, len(d["dialog"]) - self.len_context + 1, 2):
                # Combine 'len_context' consecutive utterances in a turn, and collect all turns in a list with turns
                turn = [d["dialog"][i+j].get("text","") for j in range(self.len_context)]
                utterances.append(turn)
            turns.append(utterances)

            # Also save the list with dialogue utterances
            dialogue = [(turn["id"], turn.get("text", "")) for turn in d["dialog"]]
            dialogs.append(dialogue)

            # The 'agg_persona_list" in the last utterance is the summary for the whole dialogue
            summaries.append('\n'.join(d['dialog'][-1]['agg_persona_list']))
            ids.append({"dialog_id": dialog_id, "convai_id": d["initial_data_id"]})

        return ids, turns, dialogs, summaries
        
    def __len__(self):
        return len(self.summaries)
    
    def __getitem__(self, i):
        """
        Each item is a tuple with two elements:
        0) a list of turns (each turn has two utterances, joined by '\n')
        1) a summary (which is a string of persona sentences joined by '\n')
        """

        if self.speaker_prefixes is not None:
            utterances = [
                self.speaker_prefixes[self.SELF] + t[0] + '\n' + self.speaker_prefixes[self.OTHER] + t[1]
                for t in self.turns[i]
            ]
        else:
            utterances = [
                t[0] + '\n' + t[1]
                for t in self.turns[i]                
            ]

        return utterances, self.summaries[i]

    def save_summary_fig(self, i, savedir='./'):

        dialog_id = self.indices[i]
        mapping = {"bot_0": "me", "bot_1": "you", "Nobody": "sessionbreak"}
        variant = "nopersona_nohistory"

        for j in range(0, len(self.dialogues[i]), 2):
            wrapped_turns = [(mapping[p], textwrap.wrap(t, width=45)) for p, t in self.dialogues[i][j:j+2]]
            title=f"Dataset: session_{self.session}/{self.subset}, dialog_id: {dialog_id['dialog_id']}_{j}\nvariant: {variant}"

            savepath = savedir + f"summaryfig_session_{self.session}_{self.subset}_{dialog_id['dialog_id']:06d}{j:02d}:{dialog_id['convai_id']}_{variant}"
            save_dialogue_fig(wrapped_turns, title, savepath, last_utterance_dotted=False)

    def item_measurements(self, i):
        stats = {
            "dialog_id": self.indices[i]["dialog_id"],
            "convai_id": self.indices[i]["convai_id"],
            "inputsentences": len(self[i][0]),
            "inputwords": len(' '.join(self[i][0]).split()), 
            "labelwords": len(self[i][1].split()), 
            "labelsentences": len(self[i][1].split('\n'))
        }       
        return stats

    def measurements(self):

        num_samples = len(self)
        allitem_measurements = [self.item_measurements(i) for i in range(len(self))]
        inputwords_per_sample = Counter([m["inputwords"] for m in allitem_measurements])
        labelwords_per_sample = Counter([m["labelwords"] for m in allitem_measurements])
        totalwords_per_sample = Counter([m["inputwords"] + m["labelwords"] for m in allitem_measurements])
        inputsentences_per_sample = Counter([m["inputsentences"] for m in allitem_measurements])
        labelsentences_per_sample = Counter([m["labelsentences"] for m in allitem_measurements])

        inputwords = sum([length * freq for length, freq in inputwords_per_sample.items()])
        labelwords = sum([length * freq for length, freq in labelwords_per_sample.items()])
        totalwords = sum([length * freq for length, freq in totalwords_per_sample.items()])
        inputsentences = sum([length * freq for length, freq in inputsentences_per_sample.items()])
        labelsentences = sum([length * freq for length, freq in labelsentences_per_sample.items()])

        all_measurements = {
            "allitem_measurements": allitem_measurements,
            "num_samples": num_samples,
            "inputwords": inputwords,
            "labelwords": labelwords,
            "totalwords": totalwords,
            "inputsentences": inputsentences,
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
        Transforms a list of dataset elements to list of encoded dialogue turns and the encoded persona sentences.
        Parameters:
            - data: tuple(list(string with utterance), string with summary)
        Returns:
            - tuple(list(tensor with encodes utterances)), tensor with encoded summaries
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

    def evaluate(self, model, generation_config, metrics=None, device="cpu", print_max=20, log_interval=100):

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
                    **generation_config,
                )

            preds = self.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            pred_summary = '\n'.join([pred for pred in preds if pred != self.nofact_token])

            if print_max > 0:
                print(self.formatted_item(self[i]))
                print('Prediction: ' + '\n\t' + pred_summary.replace('\n', '\n\t') + '\n')
                print_max -= 1

            target_summaries.append(target_summary)
            pred_summaries.append(pred_summary)

            if (i + 1) % log_interval == 0:
                logging.verbose(f"Evaluated {i + 1}/{len(self)} samples")
        
        stats, results_dict = calc_stats(pred_summaries, target_summaries, self.indices, metrics=metrics)

        return stats, results_dict

    @classmethod
    def predict(cls, data, model, generation_config, nofact_token='', device="cpu"):

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
                    **generation_config,
                )

            preds = cls.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            pred_summary = '\n'.join([pred for pred in preds if pred != nofact_token])

            pred_summaries.append(pred_summary)

        return pred_summaries

##
## Unit test
##

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
    parser.add_argument("--savedir", type=str, default="./output/", help="directory for output files")
    parser.add_argument("--skip_eval", default=True, action='store_true', help="just train")

    # parser = TerpMetric.add_cmdline_args(parser)
    parser = NLIMetric.add_cmdline_args(parser)
    parser = MSC_Summaries.add_cmdline_args(parser)
    parser = BartExtractor.add_cmdline_args(parser)
    args = parser.parse_args()

    # set seed
    random.seed(123)
    torch.manual_seed(args.seed)

    # prepare logging
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    # Settings for this test
    args.session = 2
    args.batch_size = 8
    subset = 'train'
    test_samples = None
    args.java_home = "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
    args.terpdir = "/Users/FrankVerhoef/Programming/terp/"
    args.tmpdir = "/Users/FrankVerhoef/Programming/PEX/output/"
    args.load = "trained_bart"
    args.speaker_prefixes = ["<other>", "<self>"]
    args.nofact_token = '' # "<nofact>"
    args.add_tokens = None #["<other>", "<self>", "<nofact>"]
    generation_config = {"num_beams": 5, "top_p": 0.9, "top_k": 10, "do_sample": True, "temperature": 1.5, "max_new_tokens": args.decoder_max}

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
        subset="test",         
    )
    m = train_set.item_measurements(0)
    train_measurements = train_set.measurements()
    logging.report(prettydict(train_measurements, title=f"Measurements for session_{args.session}/train"))

    msc_summaries = MSC_Summaries(
        basedir=args.datadir + args.basedir, 
        session=args.session, 
        subset=subset, 
        max_samples=test_samples, 
    )
    if test_samples is None:
        test_samples = len(msc_summaries)
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
    pred_summaries = msc_summaries.predict([utterances for utterances, _ in data], model, generation_config)
    logging.report(('\n----------------------------------------\n').join(pred_summaries))

    # Run evaluation
    eval_kwargs = {
        'generation_config': generation_config,
        'metrics': ["terp", "nli"], 
        'device': 'cpu', 
        'log_interval': args.log_interval, 
    }
    if "terp" in eval_kwargs['metrics']:
        TerpMetric.set(terp_dir=args.terpdir, java_home=args.java_home, tmp_dir=args.tmpdir)
    if "nli" in eval_kwargs['metrics']:
        NLIMetric.set(nli_model=args.nli_model, device=args.device, batch_size=args.batch_size)
    eval_stats, results_dict = msc_summaries.evaluate(model, **eval_kwargs)
    logging.info(eval_stats)

    # Save results
    with open(args.savedir + f"MSC_Summary_session_{msc_summaries.session}_{msc_summaries.subset}_evalresults.json", "w") as f:
        f.write(json.dumps(results_dict, sort_keys=True, indent=2))
    plot_heatmaps(results_dict, msc_summaries.session, msc_summaries.subset, savedir=args.savedir)