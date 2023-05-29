###
### Class to read the MSC summary dataset, and preprocess the data.
###

import torch
from torch.utils.data import Dataset

from torchmetrics.functional.classification import binary_confusion_matrix, binary_accuracy, binary_f1_score, binary_precision, binary_recall
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.bert import bert_score
from metrics.terp import TerpMetric

import json
import random
from collections import Counter

from utils.general import prettydict
import utils.logging as logging


class MSC_Turns(Dataset):

    tokenizer = None
    len_context = 2
    speaker_prefixes = None
    nofact_token = None
    OTHER=0
    SELF=1
    
    @classmethod
    def set(cls, tokenizer=None, len_context=2, speaker_prefixes=None, nofact_token=''):
        assert True if speaker_prefixes is None else len(speaker_prefixes) == 2, "If speaker_prefixes are set, 2 values are required"
        # assert len_context > 1, f"len_context '{len_context}' is invalid; should be at least 1"
        cls.tokenizer = tokenizer
        cls.len_context = len_context
        cls.speaker_prefixes = speaker_prefixes
        cls.nofact_token = nofact_token

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Turns')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'other' and 'self'")
        group.add_argument("--nofact_token", default='', type=str, help="Token to identify no_fact, default=''")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        group.add_argument("--len_context", default=2, type=int, help="Number of utterances to include in context")
        group.add_argument("--sessions", default=[1], nargs='+', help="MSC sessions to include in dataset")
        return parser

    def __init__(self, basedir='./', sessions=None, subset='train', max_samples=None):
        super(MSC_Turns, self).__init__()
        self.sessions = sessions
        self.subset = subset
        dialogues = []
        if sessions is not None:
            for s in self.sessions:
                filepath = f"{basedir}session_{s}/{subset}.txt"
                try:
                    with open(filepath, "r") as f:
                        for line in f:
                            dialogues.append(json.loads(line))
                except FileNotFoundError:
                    logging.warning(f"File '{filepath}' not found -> skipped")
        self.indices, self.turns, self.personas = self.transform(dialogues, max_samples)

    def transform(self, dialogues, max_samples):
        """
        Format of a summary: Dict, each dict covers one dialogue.
            - "dialog": List with Dicts, each Dict is an utterance, with corresponding information:
                - "id": String, representing the speaker
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
                - "agg_persona_list": List with Strings ==> THIS IS THE 'RUNNING' SUMMARY OF THE DIALOGUE
            - "followup": String (Sentence)
            - "newfact": String (Sentence)
            - "initial_data_id": String (id)
            - "init_personachat": Dict with the initial persona sentences
                - "init_personas": List with two lists with Strings
                    - 0: persona sentences Speaker 1
                    - 1: persona sentances Speaker 2
        """
        turns, personas, ids = [], [], []
        
        for dialog_id, d in enumerate(dialogues):
            for i in range(len(d["dialog"]) - self.len_context + 1):
                
                turn = []
                for j in range(self.len_context):
                    p = self.OTHER if (self.len_context - 1 - j) % 2 == 0  else self.SELF
                    t = d["dialog"][i+j].get("text","")
                    turn.append((p, t))
                turns.append(turn)

                if "persona_text" in d["dialog"][i+self.len_context-1].keys():
                    persona = d["dialog"][i+self.len_context-1]["persona_text"]
                else:
                    persona = self.nofact_token
                personas.append(persona)
                ids.append({"dialog_id": dialog_id, "turn_id": i, "convai_id": d["initial_data_id"]})
        
        if max_samples is not None:
            if max_samples < len(turns):
                selection = random.sample(range(len(turns)), max_samples)
                turns = [turns[i] for i in selection]
                personas = [personas[i] for i in selection]
                ids = [ids[i] for i in selection]

        return ids, turns, personas
        
    def __len__(self):
        return len(self.turns)
    
    def __getitem__(self, i):
        if self.speaker_prefixes is not None:
            history = '\n'.join([self.speaker_prefixes[p] + t for p, t in self.turns[i]])
        else:
            history = '\n'.join([t for p, t in self.turns[i]])
        return history, self.personas[i]

    def corpus(self):
        return [' '.join([*self[i]]) for i in range(len(self.turns))]

    def item_measurements(self, i):
        stats = {
            "dialog_id": self.indices[i]["dialog_id"],
            "turn_id": self.indices[i]["turn_id"],
            "convai_id": self.indices[i]["convai_id"],
            "inputwords": len(self[i][0].split()), 
            "labelwords": len(self[i][1].split()), 
        }       
        return stats

    def measurements(self):

        num_samples = len(self)
        allitem_measurements = [self.item_measurements(i) for i in range(len(self))]
        inputwords_per_sample = Counter([m["inputwords"] for m in allitem_measurements])
        labelwords_per_sample = Counter([m["labelwords"] for m in allitem_measurements])
        totalwords_per_sample = Counter([m["inputwords"] + m["labelwords"] for m in allitem_measurements])

        inputwords = sum([length * freq for length, freq in inputwords_per_sample.items()])
        labelwords = sum([length * freq for length, freq in labelwords_per_sample.items()])
        totalwords = sum([length * freq for length, freq in totalwords_per_sample.items()])

        all_measurements = {
            "allitem_measurements": allitem_measurements,
            "num_samples": num_samples,
            "inputwords": inputwords,
            "labelwords": labelwords,
            "totalwords": totalwords,
            "avg_inputwords": inputwords / num_samples,
            "avg_labelwords": labelwords / num_samples,
            "avg_totalwords": totalwords / num_samples,
            "inputwords_per_sample": sorted(inputwords_per_sample.items(), key=lambda x:x[0]),
            "labelwords_per_sample": sorted(labelwords_per_sample.items(), key=lambda x:x[0]),
            "totalwords_per_sample": sorted(totalwords_per_sample.items(), key=lambda x:x[0])
        }

        return all_measurements

    @classmethod
    def batchify(cls, data, with_labels=True, batch_format=None, batch_pad_id=0):
        """
        Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert cls.tokenizer is not None, "Missing tokenizer to batchify dataset"
        assert batch_format is not None, f"batch_format should be specified"
        assert batch_format in ["huggingface", "padded_sequences"], f"Unknown batch_format '{batch_format}' for dataset {cls.__class__.__name__}"

        history_batch, persona_batch = zip(*data)

        if batch_format == "huggingface":

            if with_labels:
                encoded = cls.tokenizer(text=history_batch, text_target=persona_batch, padding=True, return_tensors="pt")
            else:
                encoded = cls.tokenizer(text=history_batch, padding=True, return_tensors="pt")

        elif batch_format == "padded_sequences":

            if with_labels:
                ys = [torch.tensor(cls.tokenizer.encode(p).ids, dtype=torch.long) for p in persona_batch]
                ys_len = [len(y) for y in ys]
                padded_ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=batch_pad_id)
            
            xs = [torch.tensor(cls.tokenizer.encode(t).ids, dtype=torch.long) for t in history_batch]
            xs_len = [len(x) for x in xs]
            padded_xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=batch_pad_id)
            
            if with_labels:
                encoded = padded_xs, padded_ys, xs_len, ys_len
            else:
                encoded = padded_xs, xs_len

        return encoded


    def evaluate(self, model, device="cpu", decoder_max=20, batch_size=1, print_max=20, log_interval=100):

        def print_predictions(data, predictions):
            for (x, y), p in zip(data, predictions):
                print('context:    ', x)
                print('target:     ', y)
                print('prediction: ', p)
                print('-' * 40)

        model = model.to(device)
        model.eval()
        target_personas = []
        pred_personas = []
        target_facts = []
        pred_facts = []
        interval_counter = 0

        for start_index in range(0, len(self), batch_size):
            data = [self[start_index + i] for i in range(batch_size) if start_index + i < len(self)]
            batch = self.batchify(data, with_labels=False, batch_format=model.batch_format)

            with torch.no_grad():
                if model.batch_format == "huggingface":
                    pred_tokens = model.generate(
                        batch['input_ids'].to(device), 
                        max_new_tokens=decoder_max, 
                        num_beams=1,
                        do_sample=False,
                    )
                    pred_fact = model.fact_mask(pred_tokens).any(dim=1)

                elif model.batch_format == "padded_sequences":
                    pred_tokens = model.generate(batch[0].to(device), batch[2], max=decoder_max)        
                    pred_fact = pred_tokens[:, 0] != model.nofact_token_id

            pred_persona = self.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)

            pred_personas.extend(pred_persona)
            target_personas.extend([t for _, t in data])
            pred_facts.extend(pred_fact.int().tolist())
            target_facts.extend([int(t != self.nofact_token) for _, t in data])

            if print_max > 0:
                print_predictions(data, pred_persona)
                print_max -= len(data)

            interval_counter += len(data)
            if interval_counter >= log_interval:
                logging.verbose(f"Evaluated {len(pred_facts)}/{len(self)} samples")
                interval_counter =- log_interval

        stats = calc_stats_classification(pred_facts, target_facts)
        stats.update(calc_stats_generation(pred_personas, target_personas, filter_fn=lambda x:x != self.nofact_token))

        return stats


    @classmethod
    def predict_from_utterances(cls, utterances=[], model=None, device="cpu", decoder_max=20):
        assert model is not None, "No model specified to use for predictions"
        assert len(utterances) % 2 == 0, f"Received {len(utterances)} utterances, this should be an even number"
        dataset = cls()
        if len(utterances) > 0:
            dataset.turns = [[(0, utterances[i][1]), (1, utterances[i + 1][1])] for i in range(0, len(utterances), 2)]
            dataset.personas = [None for i in range(0, len(utterances), 2)]
        turns = [(dataset[i][0], "") for i in range(len(dataset))]
        pred_personas = cls.predict(turns, model, device, decoder_max)
        return pred_personas


    @classmethod
    def predict(cls, input, model, device="cpu", decoder_max=20, batch_size=1):

        model = model.to(device)
        model.eval()
        pred_personas = []

        for start_index in range(0, len(input), batch_size):
            data = [input[start_index + i] for i in range(batch_size) if start_index + i < len(input)]
            batch = cls.batchify(data, with_labels=False, batch_format=model.batch_format)

            with torch.no_grad():
                if model.batch_format == "huggingface":
                    pred_tokens = model.generate(
                        batch['input_ids'].to(device), 
                        max_new_tokens=decoder_max, 
                        num_beams=1,
                        do_sample=False,
                    )

                elif model.batch_format == "padded_sequences":
                    pred_tokens = model.generate(batch[0].to(device), batch[2], max=decoder_max)        

            pred_persona = cls.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            pred_personas.extend([p for p in pred_persona if p != cls.nofact_token])

        return pred_personas


def calc_stats_classification(pred_facts, target_facts):

    # Classification stats    
    target_facts = torch.tensor(target_facts)
    pred_facts =  torch.tensor(pred_facts)
    stats = {
        "acc": binary_accuracy(pred_facts, target_facts).item(),
        "f1": binary_f1_score(pred_facts, target_facts).item(),
        "precision": binary_precision(pred_facts, target_facts).item(),
        "recall": binary_recall(pred_facts, target_facts).item(),
        "cm": binary_confusion_matrix(pred_facts, target_facts).tolist()
    }
    return stats

def calc_stats_generation(pred_personas, target_personas, filter_fn):

    # Text generation stats; only on samples where both target and prediction comply with the filter_fn
    preds_withfact, targets_withfact = [], []
    for p, t in zip(pred_personas, target_personas):
        if filter_fn(p) and filter_fn(t):
            preds_withfact.append(p)
            targets_withfact.append(t)

    bleu_2 = bleu_score(preds_withfact, targets_withfact, n_gram=2, smooth=True).item()
    bleu_4 = bleu_score(preds_withfact, targets_withfact, n_gram=4, smooth=True).item()
    rouge_scores = rouge_score(preds_withfact, targets_withfact, rouge_keys=('rouge1', 'rouge2', 'rougeL'))
    bert_scores = bert_score(preds_withfact, targets_withfact, model_name_or_path='bert-base-uncased')
    terp_metric = TerpMetric()
    terp_metric.update(0, preds_withfact, targets_withfact)
    terp_scores = terp_metric.compute()[0]

    stats = {
        "bleu_2": bleu_2, 
        "bleu_4": bleu_4, 
        "bert_f1": sum(bert_scores['f1']) / max(len(bert_scores['f1']), 1),
        "terp": terp_scores.mean().item() if torch.numel(terp_scores) != 0 else 0
    }
    stats.update({k: v.item() for k, v in rouge_scores.items()})

    return stats

if __name__ == "__main__":

    from transformers import AutoTokenizer
    from dataset.tokenizer import train_tokenizer, PAD_TOKEN
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/'
    sessions = [2]
    subset = 'train'
    speaker_prefixes = ["<you>", "<me>"]
    nofact_token = '<nofact>'
    add_tokens = speaker_prefixes + [nofact_token]

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # tokenizer = train_tokenizer(
    #     corpus=MSC_Turns(basedir=basedir, sessions=sessions, subset='train', tokenizer=None, max_samples=1000).corpus(),
    #     max_size=4000
    # )
    # batch_pad_id = tokenizer.encode(PAD_TOKEN).ids[0]
    if add_tokens is not None:
        tokenizer.add_tokens(add_tokens)

    MSC_Turns.set(tokenizer=tokenizer, len_context=1, speaker_prefixes=speaker_prefixes, nofact_token=nofact_token)

    msc_turns = MSC_Turns(
        basedir=basedir, 
        sessions=sessions, 
        subset=subset
    )

    m = msc_turns.item_measurements(0)
    m = msc_turns.measurements()
    del m["allitem_measurements"]
    print(prettydict(m, title="Measurements"))

    data = [msc_turns[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data, batch_format="huggingface")
    # logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)
