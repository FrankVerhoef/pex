###
### Class to read the MSC session datasets, and preprocess the data.
###

import torch
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text.perplexity import Perplexity
import evaluate
from torch.utils.data import Dataset
from dataset.msc_summary_turns import MSC_Turns
import json
import random
from collections import Counter

from transformers import GenerationConfig
from dataset.convai2 import ConvAI2

import utils.logging as logging


class MSC_Session(Dataset):
    
    tokenizer = None
    speaker_prefixes = None
    
    @classmethod
    def set(cls, tokenizer, speaker_prefixes):
        assert True if speaker_prefixes is None else len(speaker_prefixes) == 2, "If speaker_prefixes are set, 2 values are required"
        cls.tokenizer = tokenizer
        cls.speaker_prefixes = {"you": speaker_prefixes[0], "me": speaker_prefixes[1]}

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Sessions')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'you' and 'me'")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        group.add_argument("--include_persona", default=False, action='store_true')
        group.add_argument("--include_history", default=False, action='store_true')
        group.add_argument("--session", default=2, type=int, help="MSC session to include in dataset")
        group.add_argument("--augmented", default=False, action='store_true', help='add all shorter versions of the dialogue to training set')
        group.add_argument("--persona_selector", type=str, default=None, help="Model to select relevant persona sentences")

        return parser

    def __init__(self, 
            basedir='./', 
            session=2, 
            subset='train', 
            include_persona=False, 
            include_history=False,
            augmented=False,
            persona_selector=None,
            max_samples=None
        ):
        super(MSC_Session, self).__init__()
        self.basedir = basedir
        self.session = session
        self.subset=subset
        self.dialogues = []
        if str(session).split(":")[0] == 'preprocessed':
            filepath = basedir + session
            self.history, self.next_utterance = self.load_preprocessed(filepath)
        else:
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
            self.include_persona = include_persona
            self.include_history = include_history
            self.augmented = augmented
            self.persona_selector = persona_selector
            self.history, self.next_utterance = self.transform_dialogues(max_samples)
            if self.persona_selector is not None:
                self.save_preprocessed()

    def load_preprocessed(self, filepath):
        logging.info("Loading preprocessed dataset from: " + filepath)
        with open(filepath, 'r') as f:
            all_history = json.loads(f.readline())
            all_next_utterance = json.loads(f.readline())
        return all_history, all_next_utterance

    def save_preprocessed(self):
        filepath = self.basedir + "preprocessed:" + f"session_{self.session}_{self.subset}"
        filepath += "_{}prefixes".format("no" if self.speaker_prefixes is None else "with")
        filepath += "_{}persona".format("selected" if self.persona_selector is not None else ("gold" if self.include_persona else "no"))
        filepath += "_{}history".format("with" if self.include_history else "no")
        logging.info("Saving preprocessed dataset to: " + filepath)
        with open(filepath, "w") as f:
            f.write(json.dumps(self.history) + '\n')
            f.write(json.dumps(self.next_utterance) + '\n') 

    def transform_dialogues(self, max_samples):
        all_history, all_next_utterance = [], []

        selected_dialogues = self.dialogues 
        if max_samples is not None:
            if max_samples < len(selected_dialogues):
                indices = random.sample(range(len(selected_dialogues)), max_samples)
                selected_dialogues = [selected_dialogues[i] for i in indices]

        for d in selected_dialogues:
            turns = d.get("dialog", [])
            personas = d.get("personas", None)  # The persona sentences ('facts') that were infered from the dialogue
            init_personas = d.get("init_personas", None)  # The initial persona sentences from ConvAI2 dataset
            previous_dialogs = d.get("previous_dialogs", None)

            self_info = {"Speaker 1": [], "Speaker 2": []} # two sets, one for each speaker
            other_info = {"Speaker 1": [], "Speaker 2": []}
            previous_utterances = []
            if self.include_persona or self.include_history:
                if previous_dialogs is not None:
                    for prev_d in previous_dialogs:
                        for i in range(len(prev_d['dialog'])):
                            text = prev_d['dialog'][i].get("text", "")
                            speaker = "Speaker 1" if i % 2 == 0 else "Speaker 2"
                            previous_utterances.append((speaker, text))

            if self.include_persona:

                if init_personas is not None:
                    # Include the persona sentences corresponding to the NEXT speaker (Speaker 1=index 0, Speaker 2=index 1)
                    self_info["Speaker 1"] = [("Speaker 1", t) for t in init_personas[0]]
                    self_info["Speaker 2"] = [("Speaker 2", t) for t in init_personas[1]]

                if self.persona_selector is None:
                    if personas is not None:
                        # Include 'gold summary' if persona_selector not defined, corresponding to the LAST speaker
                        other_info["Speaker 1"] = [("Speaker 1", t) for t in personas[0]]
                        other_info["Speaker 2"] = [("Speaker 2", t) for t in personas[1]]
                else:
                    # For Speaker 1
                    num_turns = len(previous_utterances) // 2
                    utterances = previous_utterances[:2 * num_turns]
                    other_info["Speaker 1"] = [("Speaker 1", t) for t in self.persona_selector(utterances)]

                    # For Speaker 2
                    num_turns = len(previous_utterances) // 2 - 1
                    utterances = previous_utterances[1 : 1 + 2 * num_turns]
                    other_info["Speaker 2"] = [("Speaker 2", t) for t in self.persona_selector(utterances)]

            if not self.include_history: 
                previous_utterances = []

            current_utterances = [("Speaker 1" if i % 2 == 0 else "Speaker 2", turns[i]["text"]) for i in range(len(turns))]

            start_range = 0 if self.augmented else max(len(turns) - 1, 0)
            for len_window in range(start_range, len(turns)):
                lastspeaker = "Speaker 2" if len_window < 1 else current_utterances[len_window - 1][0]
                nextspeaker = "Speaker 1" if lastspeaker == "Speaker 2" else "Speaker 2"
                history = self_info[nextspeaker] + other_info[lastspeaker] + previous_utterances + current_utterances[:len_window]
                all_history.append(history)

                next_utterance = current_utterances[len_window][1]
                all_next_utterance.append(next_utterance)

        return all_history, all_next_utterance
        
    def __len__(self):
        return len(self.history)
    
    def __getitem__(self, i):
        """
        #TODO: decide whether to include space between persona token and utterance
        """
        # other = "Speaker 2" if len(self.history[i]) < 1 else self.history[i][-1][0]
        # if other == "Speaker 2":
        if len(self.history[i]) < 1 or self.history[i][-1][0] == "Speaker 2":
            mapping = {"Speaker 1": "me", "Speaker 2": "you"}
        else:
            mapping = {"Speaker 1": "you", "Speaker 2": "me"}


        if self.speaker_prefixes is not None:
            history = ' '.join([self.speaker_prefixes[mapping[p]] + t for p, t in self.history[i]])
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

    def measurements(self):

        num_samples = self.__len__()
        inputwords = len(' '.join([self.__getitem__(i)[0] for i in range(self.__len__())]).split())
        labelwords = len(' '.join([self.__getitem__(i)[1] for i in range(self.__len__())]).split())
        avg_inputwords = inputwords / num_samples
        avg_labelwords = labelwords / num_samples
        inputwords_per_sample = Counter([len(self.__getitem__(i)[0].split()) for i in range(self.__len__())])
        inputwords_per_sample = sorted(inputwords_per_sample.items(), key=lambda x:x[0])
        labelwords_per_sample = Counter([len(self.__getitem__(i)[1].split()) for i in range(self.__len__())])
        labelwords_per_sample = sorted(labelwords_per_sample.items(), key=lambda x:x[0])
        totalwords_per_sample = Counter([
            len((self.__getitem__(i)[0] + ' ' + self.__getitem__(i)[1]).split()) 
            for i in range(self.__len__())
        ])
        totalwords_per_sample = sorted(totalwords_per_sample.items(), key=lambda x:x[0])
        avg_totalwords = sum([length * freq for length, freq in totalwords_per_sample]) / num_samples

        all_measurements = {
            "num_samples": num_samples,
            "inputwords": inputwords,
            "labelwords": labelwords,
            "avg_inputwords": avg_inputwords,
            "avg_labelwords": avg_labelwords,
            "avg_totalwords": avg_totalwords,
            "inputwords_per_sample": inputwords_per_sample,
            "labelwords_per_sample": labelwords_per_sample,
            "totalwords_per_sample": totalwords_per_sample,
        }

        return all_measurements

    @classmethod
    def batchify(cls, data, with_labels=True, batch_format=None, batch_pad_id=0, buffer=0):
        """
            Transforms a list of dataset elements to batch of consisting of contexts and a batch with the corresponding next utterance.
        """
        assert cls.tokenizer is not None, "Need to specify function to vectorize dataset"
        assert batch_format is not None, "batch_format should be specified"

        # seperate source and target sequences
        history_batch, next_utterance_batch = zip(*data)

        if batch_format == "huggingface_xycat":

            if with_labels:

                # use right padding OR LEFT ????
                # add <bos> token between context and target
                # add <eos> token at end of all targets (necessary to make sure 'shift_right' of labels is possible )
                cls.tokenizer.padding_side = 'left'
                cls.tokenizer.truncation_side = 'left'
                encoded = cls.tokenizer(
                    [history + cls.tokenizer.bos_token + next_utterance + cls.tokenizer.eos_token for history, next_utterance in data],
                    padding=True,
                    # max_length=cls.tokenizer.model_max_length, 
                    # truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

            else:

                # use left padding for the input
                cls.tokenizer.padding_side = 'left'
                cls.tokenizer.truncation_side = 'left'
                encoded = cls.tokenizer(
                    history_batch, 
                    padding=True, 
                    # max_length=cls.tokenizer.model_max_length - buffer,  # Leave some room for generation! 
                    # truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

            encoded.num_original_tokens = encoded.attention_mask.sum().item()
            encoded.num_truncated_tokens = 0
            truncated_size = encoded.input_ids.shape[1] + (buffer if not with_labels else 0) - cls.tokenizer.model_max_length
            if truncated_size > 0:
                encoded.num_truncated_tokens = encoded.attention_mask[:, :truncated_size].sum().item()
                encoded["input_ids"] = encoded.input_ids[:, truncated_size:]
                encoded["attention_mask"] = encoded.attention_mask[:, truncated_size:]
            encoded.num_tokens = encoded.attention_mask.sum().item()
            logging.spam(f"Encoded shape={encoded.input_ids.shape}, num_truncated_tokens={encoded.num_truncated_tokens}, {(encoded.num_truncated_tokens / encoded.num_original_tokens):.2%}")

        elif batch_format == "huggingface_xysplit":

            # use left padding for the input
            cls.tokenizer.padding_side = 'left'
            encoded = cls.tokenizer(
                history_batch, 
                padding=True, 
                # max_length=cls.tokenizer.model_max_length - buffer,
                # truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            if with_labels:
                
                # use right padding for labels
                # add <bos> token between context and target
                # add 'eos_token' at end of all labels (necessary to make sure 'shift_right' of labels is possible )
                cls.tokenizer.padding_side = 'right'
                labels = cls.tokenizer(
                    [cls.tokenizer.bos_token + label + cls.tokenizer.eos_token for label in next_utterance_batch],
                    padding=True, 
                    # max_length=cls.tokenizer.model_max_length,
                    # truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'            
                )

            encoded.num_original_tokens = encoded.attention_mask.sum().item()
            encoded.num_truncated_tokens = 0
            truncated_size = encoded.input_ids.shape[1] + (buffer if not with_labels else labels.input_ids.shape[1]) - cls.tokenizer.model_max_length
            if truncated_size > 0:
                encoded.num_truncated_tokens = encoded.attention_mask[:, :truncated_size].sum().item()
                encoded["input_ids"] = encoded.input_ids[:, truncated_size:]
                encoded["attention_mask"] = encoded.attention_mask[:, truncated_size:]
            encoded.num_tokens = encoded.attention_mask.sum().item()
            logging.spam(f"Encoded shape={encoded.input_ids.shape}, num_truncated_tokens={encoded.num_truncated_tokens}, {(encoded.num_truncated_tokens / encoded.num_original_tokens):.2%}")

            if with_labels:
                encoded = encoded, labels

        elif batch_format == "padded_sequences":

            if with_labels:
                ys = [torch.tensor(cls.tokenizer.encode(p).ids, dtype=torch.long) for p in next_utterance_batch]
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
        ppl = Perplexity(ignore_index=self.tokenizer.pad_token_id)
        total_original_tokens, total_truncated_tokens = 0, 0

        for start_index in range(0, self.__len__(), batch_size):
            data = [self.__getitem__(start_index + i) for i in range(batch_size) if start_index + i < self.__len__()]
            inputs, labels = self.batchify(data, with_labels=True, batch_format='huggingface_xysplit', buffer=decoder_max)
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
                        max_new_tokens=labels.input_ids.shape[1],
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                )
                # output = output.to("cpu")
            responses = self.tokenizer.batch_decode(output.sequences[:, L+1:].to("cpu")) # Do not include <bos> token in response


            if print_max > 0:
                print_responses(data, responses)
                print_max -= len(data)

            target_responses.extend([label for _, label in data])
            pred_responses.extend(responses)

            scores = torch.cat(output.scores, dim=-1).view(labels.input_ids.shape + (-1,))
            batch_ppl = ppl(scores, labels.input_ids)
            if hasattr(inputs, "num_truncated_tokens"):
                total_original_tokens += inputs.num_original_tokens
                total_truncated_tokens += inputs.num_truncated_tokens

            interval_counter += len(pred_responses)
            if interval_counter >= log_interval:
                logging.verbose(f"Evaluated {len(pred_responses)}/{self.__len__()} samples")
                interval_counter =- log_interval

        logging.info(f"Completed evaluation of {len(pred_responses)} samples")

        bleu_google = google_bleu.compute(predictions=pred_responses, references=[[t] for t in target_responses])
        meteor_score = meteor.compute(predictions=pred_responses, references=[[t] for t in target_responses])
        bleu_2 = bleu_score(pred_responses, target_responses, n_gram=2, smooth=True).item()
        bleu_4 = bleu_score(pred_responses, target_responses, n_gram=4, smooth=True).item()
        rouge_scores = rouge_score(pred_responses, target_responses, rouge_keys=('rouge1', 'rouge2', 'rougeL'))
        bert_scores = bert_score(pred_responses, target_responses, model_name_or_path='bert-base-uncased')
        perplexity = ppl.compute()
        truncation = total_truncated_tokens / max(total_original_tokens, 1)

        stats = {
            "bleu_2": bleu_2, 
            "bleu_4": bleu_4, 
            "bert_f1": sum(bert_scores['f1']) / len(bert_scores['f1']),
            "perplexity": perplexity,
            "truncation": truncation
        }
        stats.update(bleu_google)
        stats.update(meteor_score)
        stats.update({k: v.item() for k, v in rouge_scores.items()})

        return stats

if __name__ == "__main__":
    import argparse
    import random
    from functools import partial
    from transformers import AutoTokenizer
    from dataset.tokenizer import train_tokenizer, PAD_TOKEN
    from models.bart_extractor import BartExtractor

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
    checkpoint_dir = '/Users/FrankVerhoef/Programming/PEX/checkpoints/'
    subset = 'train'
    session = 4 #"preprocessed:session_2_train_withprefixes_selectedpersona_nohistory"
    if session == 1:
        version = ['both', 'revised']
        session = '-'.join(['1'] + version)
    speaker_prefixes = ['<you>', '<me>']
    add_tokens = speaker_prefixes
    include_persona = True
    include_history = True
    augmented = True
    persona_selector = None #'test_bart'

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

    if persona_selector is not None:

        # Load pretrained model to select generate (tokens for) persona sentences from a batch with input_ids
        loadpath = checkpoint_dir + persona_selector
        logging.info("Loading persona_selector from {}".format(loadpath))
        with open(loadpath + '.config', 'r') as f:
            bart_config = json.loads(f.read())
        bart_tokenizer = AutoTokenizer.from_pretrained(bart_config['bart_base'])
        if bart_config['add_tokens'] is not None:
            bart_tokenizer.add_tokens(bart_config['add_tokens'])
        bart_nofact_token_id = tokenizer.convert_tokens_to_ids(bart_config['nofact_token']) if bart_config['nofact_token'] != '' else bart_tokenizer.eos_token_id
        bart_model = BartExtractor(bart_config['bart_base'], bart_nofact_token_id)
        bart_model.bart.resize_token_embeddings(len(bart_tokenizer))
        bart_device = args.device
        if bart_device == 'mps':
            bart_device = 'cpu'
            logging.warning("Changed device from 'mps' to 'cpu' for BART persona selector")
        bart_model.load_state_dict(torch.load(loadpath, map_location=torch.device(bart_device)))

        # Configure MSC_Turns to predict persona sentences from a list of utterances
        MSC_Turns.set(
            tokenizer=bart_tokenizer, 
            len_context=2, 
            speaker_prefixes=bart_config['speaker_prefixes'], 
            nofact_token=bart_nofact_token_id
        )
        persona_selector = partial(MSC_Turns.predict_from_utterances, model=bart_model, device=bart_device)

    MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=speaker_prefixes)
    msc_turns = MSC_Session(
        basedir=datadir+basedir, 
        session=session, 
        subset=subset, 
        max_samples=2,
        include_persona=include_persona,
        include_history=include_history,
        augmented=augmented,
        persona_selector=persona_selector
    )
    
    for k,v in msc_turns.measurements.items():
        print(k, v)

    for sentence in msc_turns.corpus()[10:30]:
        logging.verbose(sentence)
    logging.verbose('-'*40)


    data = [msc_turns[i] for i in range(10)]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data, batch_format="huggingface_xycat")
    if batch_format == "huggingface_xycat":
        logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)