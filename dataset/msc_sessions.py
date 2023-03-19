###
### Class to read the MSC session datasets, and preprocess the data.
###

import torch
from torch.utils.data import Dataset
import json
import random

import utils.logging as logging


# NO_FACT_TOKEN = '<nofact>'
# NO_FACT_TOKEN = '</s>'
NO_FACT_TOKEN = ''
BATCH_FORMATS = ["huggingface", "padded_sequences"]

class MSC_Session(Dataset):
    
    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Sessions')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")
        group.add_argument("--include_persona", default=False, action='store_true')
        return parser

    def __init__(self, path, tokenizer, speaker_prefixes=None, include_persona=False, max_samples=None, batch_format="huggingface", batch_pad_id=0):
        super(MSC_Session, self).__init__()
        assert batch_format in BATCH_FORMATS, "batch_format should be one of {}".format(BATCH_FORMATS)
        assert (speaker_prefixes is None) or (len(speaker_prefixes) == 2), "Invalid number of persona prefixes ({})".format(len(speaker_prefixes))
        dialogues = []
        with open(path, "r") as f:
            for line in f:
                dialogues.append(json.loads(line))
        self.speaker_prefixes = speaker_prefixes
        self.include_persona = include_persona
        self.tokenizer = tokenizer
        self.batch_format = batch_format
        self.batch_pad_id = batch_pad_id
        self.history, self.next_utterance = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        all_history, all_next_utterance = [], []
        
        for d in dialogues:
            turns = d.get("dialog", [])
            personas = d.get("personas", None)
            num_turns = len(turns)
            if num_turns < 2:
                continue
            for len_history in range(1, num_turns):
                
                if self.include_persona and personas is not None:
                    # Include the persona sentences corresponding to the last speaker in the dialog (who has ID=0, representing the bot)
                    p = 0 if turns[len_history]["id"] == 'Speaker 1' else 1
                    history = [(0, t) for t in personas[p]]
                else:
                    history = []

                for i in range(len_history):
                    p = (len_history - i) % 2
                    t = turns[i].get("text","")
                    history.append((p, t))
                all_history.append(history)

                next_utterance = turns[len_history]["text"]

                all_next_utterance.append(next_utterance)
        
        if max_samples is not None:
            if max_samples < len(all_history):
                indices = random.sample(range(len(all_history)), max_samples)
                all_history = [all_history[i] for i in indices]
                all_next_utterance = [all_next_utterance[i] for i in indices]

        return all_history, all_next_utterance
        
    def __len__(self):
        return len(self.history)
    
    def __getitem__(self, i):
        """
        #TODO: decide whether to include space between persona token and utterance
        """
        if self.speaker_prefixes is not None:
            history = ' '.join([self.speaker_prefixes[p] + ' ' + t for p, t in self.history[i]])
            last_utterance = self.speaker_prefixes[0] + ' ' + self.next_utterance[i]
        else:
            history = ' '.join([t for p, t in self.history[i]])
            last_utterance = self.next_utterance[i]
        return history, last_utterance
    
    def corpus(self):
        return [' '.join([*(self.__getitem__(i)[:2])]) for i in range(len(self.history))]

    def batchify(self, data):
        """
            Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert self.tokenizer is not None, "Need to specify function to vectorize dataset"

        # seperate source and target sequences
        history_batch, next_utterance_batch = zip(*data)

        if self.batch_format == "huggingface":

            encoded = self.tokenizer(text=history_batch, text_target=next_utterance_batch, padding=True, return_tensors="pt")

        elif self.batch_format == "padded_sequences":

            # tokenize and convert to tensor
            xs = [torch.tensor(self.tokenizer.encode(t).ids, dtype=torch.long) for t in history_batch]
            ys = [torch.tensor(self.tokenizer.encode(p).ids, dtype=torch.long) for p in next_utterance_batch]
            
            # determine lengths of source and target
            xs_len = [len(x) for x in xs]
            ys_len = [len(y) for y in ys]

            # pad sequences
            padded_xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=self.batch_pad_id)
            padded_ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=self.batch_pad_id)
            encoded = padded_xs, padded_ys, xs_len, ys_len

        return encoded


if __name__ == "__main__":

    from transformers import AutoTokenizer
    from dataset.tokenizer import train_tokenizer
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_dialogue/session_2/train.txt'

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # tokenizer = train_tokenizer(
    #     corpus=MSC_Session(datapath, tokenizer=None, max_samples=1000).corpus(),
    #     max_size=4000
    # )

    msc_turns = MSC_Session(datapath, tokenizer, speaker_prefixes=['<self>', '<other>'], include_persona=True, batch_format="huggingface", batch_pad_id=-1)

    data = [msc_turns[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data)
    logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)