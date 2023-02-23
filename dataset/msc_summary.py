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
    
    def __init__(self, path, text2vec, len_context=2, persona_tokens=False, max_samples=None):
        super(MSC_Turns, self).__init__()
        dialogues = []
        with open(path, "r") as f:
            for line in f:
                dialogues.append(json.loads(line))
        self.persona_tokens = persona_tokens
        self.len_context = len_context
        self.text2vec = text2vec
        self.turns, self.personas = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        turns, personas = [], []
        
        for d in dialogues:
            for i in range(len(d["dialog"]) - self.len_context + 1):
                
                turn = []
                for j in range(self.len_context):
                    p = '<P{}>'.format((self.len_context - j) % 2)
                    t = d["dialog"][i+j].get("text","")
                    turn.append((p, t))
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
        if self.persona_tokens:
            turns = ' '.join([p + ' ' + t for p, t in self.turns[i]])
        else:
            turns = ' '.join([t for _, t in self.turns[i]])
        return turns, self.personas[i]
    
    def get_turn(self, i):
        return self.turns[i]

    def corpus(self):
        return [' '.join([*self.__getitem__(i)]) for i in range(len(self.turns))]

    def batchify(self, data):
        """
            Transforms a list of dataset elements to batch of consisting of dialogue turns and persona sentences.
        """
        assert self.text2vec is not None, "Need to specify function to vectorize dataset"

        # seperate source and target sequences
        turns, personas = zip(*data)

        # tokenize and convert to tensor
        xs = [torch.tensor(self.text2vec(t + END_TOKEN), dtype=torch.long) for t in turns]
        ys = [torch.tensor(self.text2vec(p + END_TOKEN), dtype=torch.long) for p in personas]
        
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
    text2vec = lambda x: random.choices(range(10), k=len(x.split()))  # return list with random integers
    msc_turns = MSC_Turns(datapath, text2vec, len_context=2)

    batch = [msc_turns[i] for i in range(10)]

    for item in batch:
        print(item[0])
        print(item[1])
        print('-'*40)

    padded_xs, padded_ys, xs_len, ys_len = msc_turns.batchify(batch)
    print(padded_xs)
    print(padded_ys)
    print(xs_len)
    print(ys_len)
    print('-' * 40)