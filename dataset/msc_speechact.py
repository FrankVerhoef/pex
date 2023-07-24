###
### Class to dataset with speech acts. These manual annotations for dialogue segments from Multi-Session Chat dataset
###

import torch
from torch.utils.data import Dataset

from torchmetrics.functional.classification import multiclass_confusion_matrix, multiclass_accuracy, multiclass_f1_score, multiclass_precision, multiclass_recall

import random
from collections import Counter

from utils.general import prettydict
import utils.logging as logging

class MSC_SpeechAct(Dataset):

    tokenizer = None
    classes = {
        'A': 'Answer',
        'E': 'Explanation',
        'G': 'Greeting',
        'O': 'Opinion',
        'P': 'Proposal',
        'Q': 'Question',
        'R': 'Reaction',
        'S': 'Statement',
    }
    id2cls = list(classes.keys())
    cls2id = {c: i for i, c in enumerate(id2cls)}
    num_classes = len(id2cls)
    
    @classmethod
    def set(cls, tokenizer=None):
        cls.tokenizer = tokenizer

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_SpeechAct')
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        return parser

    def __init__(self, basedir='./', subset='train', max_samples=None):
        super(MSC_SpeechAct, self).__init__()
        self.subset = subset
        speechacts = []
        filepath = f"{basedir}speechacts_{subset}.txt"
        try:
            with open(filepath, "r") as f:
                for line in f:
                    speechacts.append(line[:-1])    # don't copy the final '\n'
        except FileNotFoundError:
            logging.warning(f"File '{filepath}' not found -> skipped")
        self.speech, self.acts = self.transform(speechacts, max_samples)

    @classmethod
    def split_utterance(cls, s):
        s_adjusted = s.replace('St. ', 'St.').replace('Mt. ', 'Mt.')
        s_adjusted = s_adjusted.replace('  ', ' ').replace('!', '!\n').replace('? ', '?\n').replace('?" ', '?"\n').replace('. ', '.\n').replace('." ', '."\n')
        s_filtered = [s for s in s_adjusted.split('\n') if s != '' and (not s in ['.', '!', '?', ' '])]
        return s_filtered
    
    def transform(self, speechact_lines, max_samples):
        """
        Format of a speechact: string with two utterances (separated by <sep>)
        """

        speech, acts = [], []
        for i, speechact_line in enumerate(speechact_lines):
            speech_text, act_text = speechact_line.split('\t')
            turns_1, turns_2 = speech_text.split('<sep>')
            split_turns_1 = self.split_utterance(turns_1)
            split_turns_2 = self.split_utterance(turns_2)
            act_1, act_2 = act_text.split('-')
            if len(split_turns_1) == len(act_1):
                speech.extend(split_turns_1)
                acts.extend([*act_1])
            else:
                logging.warning(f"Mismatch in line {i}/1 between sentences {split_turns_1} and acts {act_1}\n{speechact_line}")
            if len(split_turns_2) == len(act_2):
                speech.extend(split_turns_2)
                acts.extend([*act_2])
            else:
                logging.warning(f"Mismatch in line {i}/2 between sentences {split_turns_2} and acts {act_2}\n{speechact_line}")
        
        if max_samples is not None:
            if max_samples < len(acts):
                selection = random.sample(range(len(acts)), max_samples)
                speech = [speech[i] for i in selection]
                acts = [acts[i] for i in selection]

        return speech, acts
        
    def __len__(self):
        return len(self.acts)
    
    def __getitem__(self, i):
        return self.speech[i], self.acts[i]

    def item_measurements(self, i):
        stats = {
            "inputwords": len(self[i][0].split()), 
        }       
        return stats

    def measurements(self):

        num_samples = len(self)
        allitem_measurements = [self.item_measurements(i) for i in range(len(self))]
        inputwords_per_sample = Counter([m["inputwords"] for m in allitem_measurements])
        inputwords = sum([length * freq for length, freq in inputwords_per_sample.items()])

        num_samples_perclass = Counter([self[i][1] for i in range(len(self))])
        num_samples_perclass = sorted(num_samples_perclass.items(), key=lambda x:x[0])
        avg_samples_perclass = [(c, n / num_samples) for c, n in num_samples_perclass]

        all_measurements = {
            "allitem_measurements": allitem_measurements,
            "num_samples": num_samples,
            "inputwords": inputwords,
            "avg_inputwords": inputwords / num_samples,
            "inputwords_per_sample": sorted(inputwords_per_sample.items(), key=lambda x:x[0]),
            "num_samples_perclass": num_samples_perclass,
            "avg_samples_perclass": avg_samples_perclass,
        }

        return all_measurements

    @classmethod
    def batchify(cls, data, with_labels=True, batch_pad_id=0):
        """
        Transforms a list of dataset elements to batch of consisting of dialogue turns and speech acts.
        """
        assert cls.tokenizer is not None, "Missing tokenizer to batchify dataset"

        speech_batch, acts_batch = zip(*data)
        encoded = cls.tokenizer(speech_batch, padding=True, return_tensors='pt')
        X = (encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids'])
        if with_labels:
            y = torch.tensor([cls.cls2id[a] for a in acts_batch], dtype=torch.long)

        return (X, y) if with_labels else X


    def evaluate(self, model, device="cpu", decoder_max=20, batch_size=1, print_max=20, log_interval=100):

        def print_predictions(data, predictions):
            for (x, y), p in zip(data, predictions):
                print('context:    ', x)
                print('target:     ', y)
                print('prediction: ', self.id2cls[p.item()])
                print('-' * 40)

        model = model.to(device)
        model.eval()
        all_labels = []
        all_preds = []
        interval_counter = 0

        for start_index in range(0, len(self), batch_size):
            data = [self[start_index + i] for i in range(batch_size) if start_index + i < len(self)]
            X, y = self.batchify(data)
            X = (x.to(device) for x in X)

            with torch.no_grad():
                logits = model(*X).cpu()
                pred = logits.argmax(dim=1)
                
            all_labels.append(y)
            all_preds.append(pred)

            if print_max > 0:
                print_predictions(data, pred)
                print_max -= len(data)

            interval_counter += len(data)
            if interval_counter >= log_interval:
                logging.verbose(f"Evaluated {len(all_preds)}/{len(self)} samples")
                interval_counter -= log_interval

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        stats = {
            "test_acc": multiclass_accuracy(all_preds, all_labels, num_classes=self.num_classes, average='weighted').item(),
            "f1": multiclass_f1_score(all_preds, all_labels, num_classes=self.num_classes, average='weighted').item(),
            "precision": multiclass_precision(all_preds, all_labels, num_classes=self.num_classes, average='weighted').item(),
            "recall": multiclass_recall(all_preds, all_labels, num_classes=self.num_classes, average='weighted').item(),
            "cm": multiclass_confusion_matrix(all_preds, all_labels, num_classes=self.num_classes).tolist()
        }

        return stats, {} # need to add result_dict


    @classmethod
    def predict(cls, input, model, device="cpu", batch_size=32):

        model = model.to(device)
        model.eval()
        all_preds = []

        single_input = isinstance(input, str)
        if single_input:
            input = [(input, "")]

        for start_index in range(0, len(input), batch_size):
            data = [(input[start_index + i], "") for i in range(batch_size) if start_index + i < len(input)]
            X = cls.batchify(data, with_labels=False)
            X = (x.to(device) for x in X)

            with torch.no_grad():
                logits = model(*X).cpu()
                pred = logits.argmax(dim=1)
                
            all_preds.extend([cls.id2cls[p] for p in pred])

        return all_preds[0] if single_input else all_preds



if __name__ == "__main__":

    from transformers import AutoTokenizer
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_speechacts/'
    subset = 'train'

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    MSC_SpeechAct.set(tokenizer=tokenizer)

    msc_speachacts = MSC_SpeechAct(
        basedir=basedir, 
        subset=subset
    )

    m = msc_speachacts.item_measurements(0)
    m = msc_speachacts.measurements()
    del m["allitem_measurements"]
    print(prettydict(m, title="Measurements"))

    data = [msc_speachacts[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_speachacts.batchify(data)
    logging.spam(batch)
