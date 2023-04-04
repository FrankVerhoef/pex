###
### Class to read the MSC summary dataset, and preprocess the data.
###


import torch
from torch.utils.data import Dataset
from torcheval.metrics.functional import bleu_score
import json
import random

import utils.logging as logging
from utils.general import padded_tensor_left

class MSC_Summaries(Dataset):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Turns')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")        
        group.add_argument("--sessions", default=[1], nargs='+', help="MSC sessions to include in dataset")
        return parser

    def __init__(self, basedir='./', sessions=[1], subset='train', tokenizer=None, speaker_prefixes=None, max_samples=None, batch_pad_id=0):
        super(MSC_Summaries, self).__init__()
        self.sessions = sessions
        self.subset = subset
        self.speaker_prefixes = speaker_prefixes
        dialogues = []
        for s in self.sessions:
            filepath = f"{basedir}session_{s}/{subset}.txt"
            try:
                with open(filepath, "r") as f:
                    for line in f:
                        dialogues.append(json.loads(line))
            except FileNotFoundError:
                logging.warning(f"File '{filepath}' not found -> skipped")
        self.tokenizer = tokenizer
        self.batch_pad_id = batch_pad_id
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

    def batchify(self, data):
        """
        Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert self.tokenizer is not None, "Need to specify function to vectorize dataset"
        assert self.tokenizer.padding_side == 'left', "Tokenizer padding_side must be 'left'"

        # seperate source and target sequences
        utterances, summaries = zip(*data)

        encoded_utterances = [
            self.tokenizer(text=turns, padding=True, return_tensors="pt")
            for turns in utterances
        ]
        encoded_summaries = padded_tensor_left(
            [
                self.tokenizer.encode(text=s, padding=False, return_tensors="pt")[0]
                for s in summaries
            ], 
            pad_value=self.batch_pad_id
        )

        return encoded_utterances, encoded_summaries


    def evaluate(self, model, device="cpu", decoder_max=20, print_max=20, log_interval=100):

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
                    num_beams=1,
                    do_sample=False,
                )

            pred_summary = '\n'.join(self.tokenizer.batch_decode(pred_tokens.cpu().tolist(), skip_special_tokens=True))

            if print_max > 0:
                print_predictions(self.__getitem__(i), pred_summary)
                print_max -= 1

            target_summaries.append(target_summary)
            pred_summaries.append(pred_summary)

            if (i + 1) % log_interval == 0:
                logging.verbose(f"Evaluated {i + 1}/{self.__len__()} samples")
        
        try:
            bleu_4 = bleu_score(pred_summaries, target_summaries).item()
        except ValueError:
            bleu_4 = 0

        stats = {
            "bleu": bleu_4
        }

        return stats

if __name__ == "__main__":

    from transformers import AutoTokenizer

    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/session_1/train.txt'
    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/'
    sessions = [1]
    subset = 'train'
    speaker_prefixes = ["<self>", "<other>"]
    add_tokens = speaker_prefixes

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", padding_side='left')
    tokenizer.add_tokens(add_tokens)

    msc_summaries = MSC_Summaries(
        basedir=basedir, 
        sessions=sessions, 
        subset=subset, 
        tokenizer=tokenizer, 
        speaker_prefixes=speaker_prefixes, 
        batch_pad_id=tokenizer.pad_token_id
    )
    data = [msc_summaries[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_summaries.batchify(data)
    # logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)