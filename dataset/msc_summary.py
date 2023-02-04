###
### Class to read the MSC summary dataset, and preprocess the data.
###

import torch
from torch.utils.data import Dataset
import json
import random

from dataset.vocab import END_TOKEN, PAD_TOKEN


extra_tokens = ['<P0>', '<P1>']

class MSC_Turns(Dataset):
    
    def __init__(self, path, text2vec, len_context=2, max_samples=None):
        super(MSC_Turns, self).__init__()
        dialogues = []
        with open(path, "r") as f:
            for line in f:
                dialogues.append(json.loads(line))
        self.len_context = len_context
        self.text2vec = text2vec
        self.turns, self.personas = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        turns, personas = [], []
        
        for d in dialogues:
            for i in range(len(d["dialog"]) - self.len_context + 1):
                
                turn = ""
                for j in range(self.len_context):
                    turn += '<P{}> '.format((self.len_context - j) % 2)
                    turn += d["dialog"][i+j].get("text","") + ' '
                turns.append(turn)

                if "persona_text" in d["dialog"][i+self.len_context-1].keys():
                    persona = d["dialog"][i+self.len_context-1]["persona_text"] + ' '
                else:
                    persona = ''
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
        return self.turns[i], self.personas[i]

    def corpus(self):
        return [' '.join([turn, persona]) for turn, persona in zip(self.turns, self.personas)]

    def batchify(self, data):
        """
            Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert self.text2vec is not None, "Need to specify function to vectorize dataset"

        # seperate source and target sequences
        turns, personas = zip(*data)

        # tokenize and convert to tensor
        xs = [torch.tensor(self.text2vec(t + END_TOKEN)) for t in turns]
        ys = [torch.tensor(self.text2vec(p + END_TOKEN)) for p in personas]

        # determine lengths of source and target
        xs_len = [len(x) for x in xs]
        ys_len = [len(y) for y in ys]

        # pad sequences
        pad_value = self.text2vec(PAD_TOKEN)[0]
        padded_xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_value)
        padded_ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_value)
        
        return padded_xs, padded_ys, xs_len, ys_len


if __name__ == "__main__":
    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/session_1/train.txt'

    # Test extraction of dialogue turns and persona sentences
    msc_turns = MSC_Turns(datapath, text2vec=None, len_context=2)

    batch = [msc_turns[i] for i in range(10)]

    for item in batch:
        print(item[0])
        print(item[1])
        print('-'*40)

    print(msc_turns.batchify(batch))
    print('-' * 40)