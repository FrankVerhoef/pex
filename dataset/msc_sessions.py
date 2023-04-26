###
### Class to read the MSC session datasets, and preprocess the data.
###

import torch
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
import evaluate
from torch.utils.data import Dataset
from dataset.msc_summary_turns import MSC_Turns
import json
import random

from transformers import GenerationConfig
from dataset.convai2 import ConvAI2

import utils.logging as logging


BATCH_FORMATS = ["huggingface_xycat", "huggingface_xysplit", "padded_sequences"]

class MSC_Session(Dataset):
    
    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Sessions')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'self' and 'other'")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        group.add_argument("--include_persona", default=False, action='store_true')
        group.add_argument("--include_history", default=False, action='store_true')
        group.add_argument("--session", default=2, type=int, help="MSC session to include in dataset")
        group.add_argument("--augmented", default=False, action='store_true', help='add all shorter versions of the dialogue to training set')
        return parser

    def __init__(self, 
            basedir='./', 
            session=2, 
            subset='train', 
            tokenizer=None, 
            speaker_prefixes=None, 
            include_persona=False, 
            include_history=False,
            augmented=False,
            persona_selector=None,
            max_samples=None, 
            batch_format="huggingface_xycat", 
            batch_pad_id=0, 
            **kwargs
        ):
        super(MSC_Session, self).__init__()
        assert batch_format in BATCH_FORMATS, "batch_format should be one of {}".format(BATCH_FORMATS)
        assert True if speaker_prefixes is None else len(speaker_prefixes) == 2, "Invalid number of speaker prefixes ({})".format(len(speaker_prefixes))
        self.session = session
        self.subset=subset
        self.dialogues = []
        if str(session)[0] == '1':
            version = str(session).split('-')[1:]
            if len(version) > 0:
                convai2_dataset = ConvAI2(basedir=basedir + 'ConvAI2/', version=version, subset=subset)
            else:
                convai2_dataset = ConvAI2(basedir=basedir + 'ConvAI2/', subset=subset)
            logging.info(f"Read {len(convai2_dataset)} dialogues from ConvAI2 for {subset} dataset")
            self.dialogues.extend([convai2_dataset[i] for i in range(len(convai2_dataset))])
        else:
            filepath = f"{basedir}session_{session}/{subset}.txt"
            try:
                with open(filepath, "r") as f:
                    msc_dialogues = [json.loads(line) for line in f]
                logging.info(f"Read {len(msc_dialogues)} dialogues from MSC session {session} for {subset} dataset")
                self.dialogues.extend(msc_dialogues)
            except FileNotFoundError:
                logging.warning(f"File '{filepath}' not found -> skipped")
        self.speaker_prefixes = speaker_prefixes
        self.include_persona = include_persona
        self.include_history = include_history
        self.augmented = augmented
        self.persona_selector = persona_selector
        self.tokenizer = tokenizer
        self.batch_format = batch_format
        self.batch_pad_id = batch_pad_id
        self.history, self.next_utterance = self.transform_dialogues(max_samples)   

    def transform_dialogues(self, max_samples):
        all_history, all_next_utterance = [], []
        
        for d in self.dialogues:
            turns = d.get("dialog", [])
            personas = d.get("personas", None)  # The persona sentences ('facts') that were infered from the dialogue
            init_personas = d.get("init_personas", None)  # The initial persona sentences from ConvAI2 dataset
            previous_dialogs = d.get("previous_dialogs", None)
            start_range = 0 if self.augmented else max(len(turns) - 1, 0)
            for len_history in range(start_range, len(turns)):
                history = []
                previous_utterances = []
                if self.include_persona or self.include_history:
                    if previous_dialogs is not None:
                        for prev_d in previous_dialogs:
                            for i in range(len(prev_d['dialog'])):
                                p = (len_history - i) % 2
                                t = prev_d['dialog'][i].get("text", "")
                                previous_utterances.append((p, t))

                if self.include_persona:
                    if self.persona_selector is None:
                        if personas is not None:
                            # Include summary of persona sentences corresponding to the last speaker -> 'gold summary'
                            p_hist = 0 if turns[len_history - 1]["id"] == 'Speaker 1' else 1
                            history.extend([(1, t) for t in personas[p_hist]])
                    else:
                        firstspeaker_id = len_history % 2 #TODO: THINK ABOUT THIS!
                        batch = MSC_Turns.batch_from_utterances(previous_utterances, firstspeaker_id)
                        history.extend([(1, t) for t in self.persona_selector(batch)])

                    if init_personas is not None:
                        # Include the persona sentences corresponding to the next speaker (Speaker 1=index 0, Speaker 2=index 1)
                        p_next = 0 if turns[len_history]["id"] == 'Speaker 1' else 1
                        history.extend([(0, t) for t in init_personas[p_next]])

                if self.include_history:
                    history.extend(previous_utterances)

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
            history = ' '.join([self.speaker_prefixes[p] + t for p, t in self.history[i]])
            next_utterance = self.next_utterance[i] # Do not include speaker prefix for target
        else:
            history = ' '.join([t for _, t in self.history[i]])
            next_utterance = self.next_utterance[i]
        return history, next_utterance
    
    def corpus(self):
        corpus = []
        for dialog in self.dialogues:
            for personas in dialog.get("personas", []):
                corpus.extend(personas)
            for utterance in dialog.get("dialog", []):
                corpus.append(utterance['text'])
        return corpus

    def batchify(self, data):
        """
            Transforms a list of dataset elements to batch of consisting of contexts and a batch with the corresponding next utterance.
        """
        assert self.tokenizer is not None, "Need to specify function to vectorize dataset"

        # seperate source and target sequences
        history_batch, next_utterance_batch = zip(*data)

        if self.batch_format == "huggingface_xycat":

            # use right padding
            # add <bos> token between context and target
            # add <eos> token at end of all targets (necessary to make sure 'shift_right' of labels is possible )
            self.tokenizer.padding_side = 'right'
            self.tokenizer.truncation_side = 'left'
            encoded = self.tokenizer(
                [history + self.tokenizer.bos_token + next_utterance + self.tokenizer.eos_token for history, next_utterance in data],
                padding=True,
                max_length=self.tokenizer.model_max_length, 
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'            
            )
            logging.spam(f"Encoded shape={encoded.input_ids.shape}")

        elif self.batch_format == "huggingface_x":

            # use left padding for the input
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            encoded = self.tokenizer(
                history_batch, 
                padding=True, 
                max_length=self.tokenizer.model_max_length - 50,  # Leave some room for generation! 
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )            

        elif self.batch_format == "huggingface_xysplit":

            # use left padding for the input
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                history_batch, 
                padding=True, 
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            # use right padding for labels
            # add <bos> token between context and target
            # add 'eos_token' at end of all labels (necessary to make sure 'shift_right' of labels is possible )
            self.tokenizer.padding_side = 'right'
            labels = self.tokenizer(
                [self.tokenizer.bos_token + label + self.tokenizer.eos_token for label in next_utterance_batch],
                padding=True, 
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'            
            )
            encoded = inputs, labels

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


    def evaluate(self, model, device="cpu", decoder_max=20, batch_size=1, print_max=20, log_interval=100):

        def print_responses(data, responses):
            for (x, y), p in zip(data, responses):
                print('context:    ', x)
                print('target:     ', y)
                print('prediction: ', p)
                print('-' * 40)

        model = model.to(device)
        model.eval()
        target_responses = []
        pred_responses = []
        interval_counter = 0
        meteor = evaluate.load("meteor")
        google_bleu = evaluate.load("google_bleu")

        for start_index in range(0, self.__len__(), batch_size):
            data = [self.__getitem__(start_index + i) for i in range(batch_size) if start_index + i < self.__len__()]
            inputs = self.batchify(data)
            B, L = inputs.input_ids.shape[:2]
            bos_tokens = torch.full((B, 1), fill_value=model.bos_token_id, dtype=torch.long, device=inputs.input_ids.device)

            with torch.no_grad():
                output = model.model.generate(
                    # Add the bos_token to the input. 
                    inputs=torch.cat([inputs.input_ids, bos_tokens], dim=1).to(device), 
                    # inputs.input_ids,
                    generation_config=GenerationConfig(
                        pad_token_id=model.model.config.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=decoder_max
                    )
                )
                output = output.cpu()
            responses = self.tokenizer.batch_decode(output[:, L+1:]) # Do not include <bos> token in response

            if print_max > 0:
                print_responses(data, responses)
                print_max -= len(data)

            target_responses.extend([label for _, label in data])
            pred_responses.extend(responses)

            interval_counter += len(pred_responses)
            if interval_counter >= log_interval:
                logging.verbose(f"Evaluated {len(pred_responses)}/{self.__len__()} samples")
                interval_counter =- log_interval

        logging.info(f"Completed evaluation of {len(pred_responses)} samples")

        bleu_google = google_bleu.compute(predictions=pred_responses, references=[[t] for t in target_responses])
        meteor_score = meteor.compute(predictions=pred_responses, references=[[t] for t in target_responses])
        bleu_2 = bleu_score(target_responses, pred_responses, n_gram=2, smooth=True).item()
        bleu_4 = bleu_score(target_responses, pred_responses, n_gram=4, smooth=True).item()
        rouge_scores = rouge_score(pred_responses, target_responses, rouge_keys=('rouge1', 'rouge2', 'rougeL'))

        stats = {
            "bleu_2": bleu_2, 
            "bleu_4": bleu_4, 
            "gleu": bleu_google, 
            "meteor": meteor_score
        }
        stats.update({k: v.item() for k, v in rouge_scores.items()})

        return stats

if __name__ == "__main__":
    import argparse
    import random
    from transformers import AutoTokenizer
    from dataset.tokenizer import train_tokenizer, PAD_TOKEN

    def get_parser():

        parser = argparse.ArgumentParser(description="Train a KnowledgeGroundedDecoder", conflict_handler='resolve')

        # General, loading, saving, logging
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--loglevel", type=str, default="SPAM", choices=logging.get_all_levels())
        parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
        
        # Training
        parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

        return parser

    parser = get_parser()
    args = parser.parse_known_args()[0]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    args = parser.parse_args()

    datadir = '/Users/FrankVerhoef/Programming/PEX/data/'
    basedir = 'msc/msc_dialogue/'
    subset = 'train'
    session = 2
    if session == 1:
        version = ['both', 'revised']
        session = '-'.join(['1'] + version)
    speaker_prefixes = ['<me>', '<you>']
    add_tokens = speaker_prefixes
    include_persona = False
    include_history = False
    augmented = False

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if add_tokens is not None:
        tokenizer.add_tokens(add_tokens)
    tokenizer.bos_token_id = tokenizer.eos_token_id
    if speaker_prefixes is not None:
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(speaker_prefixes[0]) # Will return eos_token_id, unless <self> token had been added to tokenizer

    batch_format = "huggingface_xycat"
    # tokenizer = train_tokenizer(
    #     corpus=MSC_Session(basedir=datadir + basedir, session=session, tokenizer=None, max_samples=1000).corpus(),
    #     max_size=4000
    # )
    # pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
    # batch_format = "padded_sequences"

    msc_turns = MSC_Session(
        basedir=datadir+basedir, 
        session=session, 
        subset=subset, 
        tokenizer=tokenizer, 
        speaker_prefixes=speaker_prefixes,
        include_persona=include_persona,
        include_history=include_history,
        augmented=augmented,
        batch_pad_id=tokenizer.pad_token_id,
        batch_format=batch_format
    )
    for sentence in msc_turns.corpus()[10:30]:
        logging.verbose(sentence)
    logging.verbose('-'*40)
        
    data = [msc_turns[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data)
    if batch_format == "huggingface":
        logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)