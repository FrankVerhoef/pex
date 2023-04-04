###
### Class to read the MSC summary dataset, and preprocess the data.
###

import torch
from torch.utils.data import Dataset
from torcheval.metrics.functional import bleu_score, binary_accuracy, binary_f1_score, binary_confusion_matrix
import json
import random

import utils.logging as logging

BATCH_FORMATS = ["huggingface", "padded_sequences"]

class MSC_Turns(Dataset):
    
    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Turns')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")
        group.add_argument("--nofact_token", default='', type=str, help="Token to identify no_fact, default=''")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        group.add_argument("--len_context", default=2, type=int, help="Number of utterances to include in context")
        group.add_argument("--sessions", default=[1], nargs='+', help="MSC sessions to include in dataset")
        return parser

    def __init__(self, basedir='./', sessions=[1], subset='train', tokenizer=None, len_context=2, speaker_prefixes=None, nofact_token='', max_samples=None, batch_format="huggingface", batch_pad_id=0):
        super(MSC_Turns, self).__init__()
        assert batch_format in BATCH_FORMATS, f"batch_format '{batch_format}' is invalid; should be one of {BATCH_FORMATS}"
        assert len_context > 1, f"len_context '{len_context}' is invalid; should be at least 1"
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
        self.tokenizer = tokenizer
        self.len_context = len_context
        self.speaker_prefixes = speaker_prefixes
        self.nofact_token = nofact_token
        self.batch_format = batch_format
        self.batch_pad_id = batch_pad_id
        self.turns, self.personas = self.transform(dialogues, max_samples)
        
    def transform(self, dialogues, max_samples):
        turns, personas = [], []
        
        for d in dialogues:
            for i in range(len(d["dialog"]) - self.len_context + 1):
                
                turn = []
                for j in range(self.len_context):
                    p = (self.len_context - j) % 2
                    t = d["dialog"][i+j].get("text","")
                    turn.append((p, t))
                turns.append(turn)

                if "persona_text" in d["dialog"][i+self.len_context-1].keys():
                    persona = d["dialog"][i+self.len_context-1]["persona_text"]
                else:
                    persona = self.nofact_token
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

        if self.speaker_prefixes is not None:
            history = ' '.join([self.speaker_prefixes[p] + ' ' + t for p, t in self.turns[i][:-1]])
            last_utterance = self.speaker_prefixes[last_p] + ' ' + last_t
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


    def evaluate(self, model, device="cpu", decoder_max=20, print_max=20, log_interval=100):

        def print_predictions(text_in, text_out):

            x, y = text_in
            print('context:    ', x)
            print('target:     ', y)
            print('prediction: ', text_out)
            print('-' * 40)

        model = model.to(device)
        model.eval()
        target_personas = []
        pred_personas = []
        target_facts = []
        pred_facts = []

        for i in range(self.__len__()):

            target_persona = self.__getitem__(i)[1]
            batch = self.batchify([self.__getitem__(i)])  # Batch with one sample

            with torch.no_grad():
                if self.batch_format == "huggingface":
                    pred_tokens = model.generate(
                        batch['input_ids'].to(device), 
                        min_length=2,
                        max_new_tokens=decoder_max, 
                        num_beams=1,
                        do_sample=False,
                    )[0]
                    pred_fact = pred_tokens[2] != model.nofact_token_id

                elif self.batch_format == "padded_sequences":
                    pred_tokens = model.generate(batch[0].to(device), batch[2], max=decoder_max)[0]              
                    pred_fact = pred_tokens[0] != model.nofact_token_id

            if pred_fact:
                pred_persona = self.tokenizer.decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            else:
                pred_persona = self.nofact_token

            if print_max > 0:
                print_predictions(self.__getitem__(i), pred_persona)
                print_max -= 1

            if target_persona != self.nofact_token:
                target_facts.append(1)
                target_personas.append(target_persona)
                pred_personas.append(pred_persona)
            else:
                target_facts.append(0)
            pred_facts.append(pred_fact.int().item())

            if (i + 1) % log_interval == 0:
                logging.verbose(f"Evaluated {i}/{self.__len__()} samples")

        target_facts = torch.tensor(target_facts)
        pred_facts =  torch.tensor(pred_facts)
        
        try:
            bleu_4 = bleu_score(pred_personas, target_personas).item()
        except ValueError:
            bleu_4 = 0

        stats = {
            "test_acc": binary_accuracy(pred_facts, target_facts).item(),
            "f1": binary_f1_score(pred_facts, target_facts).item(),
            "cm": binary_confusion_matrix(pred_facts, target_facts).tolist(),
            "bleu": bleu_4
        }

        return stats

if __name__ == "__main__":

    from transformers import AutoTokenizer
    from dataset.tokenizer import train_tokenizer
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/session_1/train.txt'
    speaker_prefixes = ["<me>", "<you>"]
    nofact_token = '<nofact>'
    add_tokens = speaker_prefixes + [nofact_token]

    # Test extraction of dialogue turns and persona sentences
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    tokenizer = train_tokenizer(
        corpus=MSC_Turns(datapath, tokenizer=None, max_samples=1000).corpus(),
        max_size=4000
    )
    tokenizer.add_tokens(add_tokens)

    msc_turns = MSC_Turns(datapath, tokenizer, len_context=3, speaker_prefixes=speaker_prefixes, nofact_token=nofact_token, batch_format="padded_sequences", batch_pad_id=-1)

    data = [msc_turns[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data)
    # logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)