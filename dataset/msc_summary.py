###
### Class to read the MSC summary dataset, and preprocess the data.
###

import torch
from torch.utils.data import Dataset
import json
import random

import utils.logging as logging

persona_token = {
    0: '<self>',
    1: '<other>'
}
PERSONA_TOKENS = list(persona_token.values())

persona_prefix = {
    0: "<self>",
    1: "<other>"
}

# NO_FACT_TOKEN = '<nofact>'
# NO_FACT_TOKEN = '</s>'
NO_FACT_TOKEN = ''
BATCH_FORMATS = ["huggingface", "padded_sequences"]

class MSC_Turns(Dataset):
    
    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Turns')
        group.add_argument("--persona_identifier", type=str, default=None, choices=[None, "token", "text"], help="Whether to insert persona identifier before each dialogue turn")
        return parser

    def __init__(self, path, tokenizer, len_context=2, persona_identifier=None, max_samples=None, batch_format="huggingface", batch_pad_id=0):
        super(MSC_Turns, self).__init__()
        assert batch_format in BATCH_FORMATS, "batch_format should be one of {}".format(BATCH_FORMATS)
        dialogues = []
        with open(path, "r") as f:
            for line in f:
                dialogues.append(json.loads(line))
        self.persona_tokens = persona_identifier == "token"
        self.persona_prefix = persona_identifier == "text"
        self.len_context = len_context
        self.tokenizer = tokenizer
        self.batch_format = batch_format
        self.batch_pad_id = batch_pad_id
        self.turns, self.personas = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        turns, personas = [], []
        
        for d in dialogues:
            for i in range(len(d["dialog"]) - self.len_context + 1):
                
                turn = []
                for j in range(self.len_context):
                    # p = '<P{}>'.format((self.len_context - j) % 2)
                    p = (self.len_context - j) % 2
                    t = d["dialog"][i+j].get("text","")
                    turn.append((p, t))
                turns.append(turn)

                if "persona_text" in d["dialog"][i+self.len_context-1].keys():
                    persona = d["dialog"][i+self.len_context-1]["persona_text"] + ' '
                else:
                    persona = NO_FACT_TOKEN
                personas.append(persona)
        
        if max_samples is not None:
            if max_samples < len(turns):
                indices = random.sample(range(len(turns)), max_samples)
                turns = [turns[i] for i in indices]
                personas = [personas[i] for i in indices]

        return turns, personas
        
    def __len__(self):
        return len(self.turns)
    
    def __getitem__(self, i):
        """
        #TODO: decide whether to include space between persona token and utterance
        """
        last_p, last_t = self.turns[i][-1]
        if self.persona_tokens:
            history = ' '.join([persona_token[p] + ' ' + t for p, t in self.turns[i][:-1]])
            last_utterance = persona_token[last_p] + ' ' + last_t
        elif self.persona_prefix:
            history = ' '.join([persona_prefix[p] + ' ' + t for p, t in self.turns[i][:-1]])
            last_utterance = persona_prefix[last_p] + ' ' + last_t
        else:
            history = ' '.join([t for p, t in self.turns[i][:-1]])
            last_utterance = last_t
        return history + ' ' + last_utterance, self.personas[i]
    
    def corpus(self):
        return [' '.join([*self.__getitem__(i)]) for i in range(len(self.turns))]

    def batchify(self, data):
        """
            Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert self.tokenizer is not None, "Need to specify function to vectorize dataset"

        # seperate source and target sequences
        utterances, personas = zip(*data)

        if self.batch_format == "huggingface":

            encoded = self.tokenizer(text=utterances, text_target=personas, padding=True, return_tensors="pt")

        elif self.batch_format == "padded_sequences":

            # tokenize and convert to tensor
            xs = [torch.tensor(self.tokenizer.encode(t).ids, dtype=torch.long) for t in utterances]
            ys = [torch.tensor(self.tokenizer.encode(p).ids, dtype=torch.long) for p in personas]
            
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

    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/session_1/train.txt'

    # Test extraction of dialogue turns and persona sentences
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    tokenizer = train_tokenizer(
        corpus=MSC_Turns(datapath, tokenizer=None, max_samples=1000).corpus(),
        max_size=4000
    )

    msc_turns = MSC_Turns(datapath, tokenizer, len_context=3, persona_identifier="token", batch_format="padded_sequences", batch_pad_id=-1)

    data = [msc_turns[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data)
    # logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)