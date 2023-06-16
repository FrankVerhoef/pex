###
### Class to read the MSC session datasets, and preprocess the data.
###

import torch

from torchmetrics import SacreBLEUScore, BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
import evaluate

from torch.utils.data import Dataset
from dataset.msc_summary_turns import MSC_Turns
import json
import random
from datetime import datetime
import textwrap
from collections import Counter

from transformers import GenerationConfig, LogitsProcessorList, NoRepeatNGramLogitsProcessor
from dataset.convai2 import ConvAI2

import utils.logging as logging
from utils.general import prettydict
from utils.plotting import save_dialogue_fig

INPUT_ORDER_OPTIONS = ['personas-history-current', 'history-personas-current']

class MSC_Metrics:

    def __init__(self, ignore_index, device):
        self.indices = []
        self.responses = []
        self.targets = []
        self.perc_truncated_tokens = torch.tensor([])
        self.sacreblue4_score = SacreBLEUScore(n_gram=4)
        self.bleu2_score = BLEUScore(n_gram=2, smooth=True)
        self.bleu4_score = BLEUScore(n_gram=4, smooth=True)
        self.rouge_score = ROUGEScore(rouge_keys='rougeL')
        self.bert_score = BERTScore(model_name_or_path='bert-base-uncased')
        self.meteor = evaluate.load("meteor", experiment_id=datetime.now().strftime("%j%H%M%S"))
        self.google_bleu = evaluate.load("google_bleu")

    def update(self, responses, targets, input_batch, label_batch, output_batch, indices):
    
        self.indices.extend(indices)
        self.responses.extend(responses)
        self.targets.extend(targets)
        if hasattr(input_batch, "num_truncated_tokens"):
            avg_truncation = torch.div(
                input_batch.num_truncated_tokens.float(), 
                torch.maximum(input_batch.num_original_tokens, torch.ones_like(input_batch.num_original_tokens)),
            )
            self.perc_truncated_tokens = torch.cat([self.perc_truncated_tokens, avg_truncation], dim=0)
        self.sacreblue4_score.update(responses, [[t] for t in targets])
        self.bleu2_score.update(responses, [[t] for t in targets])
        self.bleu4_score.update(responses, [[t] for t in targets])
        self.rouge_score.update(responses, targets)
        self.bert_score.update(responses, targets)
        self.meteor.add_batch(predictions=responses, references=targets)
        self.google_bleu.add_batch(predictions=responses, references=targets)

    def compute(self):

        def turn_id(id):
            return id["session"], id['dialog_id'], id['turn_id']

        rouge_scores = self.rouge_score.compute()
        bert_scores = self.bert_score.compute()

        result_dict = {
            turn_id(id): {
                'input_truncation': self.perc_truncated_tokens[i].item(),
                'pred_response': self.responses[i],
                'target_response': self.targets[i],
                "bert_f1": bert_scores['f1'][i],
                "bert_precision": bert_scores['precision'][i],
                "bert_recall": bert_scores['recall'][i]
            }
            for i, id in enumerate(self.indices)
        }

        stats = {
            "truncation": self.perc_truncated_tokens.mean().item() if len(self.perc_truncated_tokens) > 0 else 0,
            "sacreblue_4": self.sacreblue4_score.compute().item(),
            "bleu_2": self.bleu2_score.compute().item(), 
            "bleu_4": self.bleu4_score.compute().item(),             
        }
        stats['bert_f1'] = sum(bert_scores['f1']) / len(bert_scores['f1'])
        stats.update({k: v.item() for k, v in rouge_scores.items()})
        stats.update(self.meteor.compute())
        stats.update(self.google_bleu.compute())

        return stats, result_dict
    

class MSC_Session(Dataset):
    
    tokenizer = None
    speaker_prefixes = None
    sessionbreak_token = None
    
    @classmethod
    def set(cls, tokenizer=None, speaker_prefixes=None, sessionbreak_token=None):
        assert True if speaker_prefixes is None else len(speaker_prefixes) == 2, "If speaker_prefixes are set, 2 values are required"
        cls.tokenizer = tokenizer
        cls.sessionbreak_token = sessionbreak_token
        if speaker_prefixes is not None:
            cls.speaker_prefixes = {"you": speaker_prefixes[0], "me": speaker_prefixes[1]}
            if sessionbreak_token is not None:
                cls.speaker_prefixes["sessionbreak"] = sessionbreak_token


    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Sessions')
        group.add_argument("--speaker_prefixes", default=None, nargs=2, help="prefixes for 'you' and 'me'")
        group.add_argument("--add_tokens", default=None, nargs='*', help="Tokens to add to tokenizer")
        group.add_argument("--include_persona", default=False, action='store_true')
        group.add_argument("--include_history", default=False, action='store_true')
        group.add_argument("--input_order", default='personas-history-current', choices=INPUT_ORDER_OPTIONS)
        group.add_argument("--sessionbreak_token", type=str, default=None, help="Token to insert to mark separation between dialogue sessions")
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
            input_order='personas-history-current',
            augmented=False,
            persona_selector=None,
            persona_selector_fn=None,
            max_samples=None
        ):
        super(MSC_Session, self).__init__()
        assert input_order in INPUT_ORDER_OPTIONS, f"input_order should be one of {INPUT_ORDER_OPTIONS}"
        self.basedir = basedir
        self.session = session
        self.subset=subset
        self.include_persona = include_persona
        self.include_history = include_history
        self.input_order = input_order
        self.augmented = augmented
        self.persona_selector = persona_selector
        self.persona_selector_fn = persona_selector_fn
        self.max_samples = max_samples
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
        self.indices, self.history, self.next_utterance = self.transform_dialogues(max_samples)


    def load_preprocessed(self):
        filepath = f"{self.basedir}{self.persona_selector}:session_{self.session}_{self.subset}.json"
        logging.info("Loading preprocessed summaries from: " + filepath)
        try:
            with open(filepath, 'r') as f:
                preprocessed = {int(k) : v for k, v in json.loads(f.read()).items()}
        except:
            modelname = self.persona_selector.split(':')[1]
            logging.warning(f"Error opening file: {filepath}, using model {modelname} to extract persona sentences from dialogue history")
            preprocessed = {}
        return preprocessed

    def save_preprocessed(self, preprocessed):
        if self.persona_selector.split(':')[0] == 'preprocessed':
            modelname = self.persona_selector.split(':')[1]
        else:
            modelname = self.persona_selector
        filepath = f"{self.basedir}preprocessed:{modelname}:session_{self.session}_{self.subset}.json"
        logging.info("Saving extracted persona sentences to: " + filepath)
        with open(filepath, "w") as f:
            f.write(json.dumps(preprocessed, indent=2))


    def transform_dialogues(self, max_samples):
        """
        Format of a session: Dict, each dict covers one dialogue.
            - "personas": List with two lists with Strings ==> THIS IS THE 'RUNNING' SUMMARY OF THE DIALOGUE
                - persona sentences Speaker 1
                - persona sentences Speaker 2
            - "dialog": List with Dicts, each Dict is an utterance, with corresponding information:
                - "id": String, representing the speaker ('Speaker 1' or 'Speaker 2')
                - "text": String
                - "convai_id": String
                - "agg_persona_list": List with Strings
            - "metadata": Dict with:
                - "initial_data_id": String (id)
                - "session_id": Int
            - "previous_dialogs": 
                - "personas":
                - "dialog": List of Dicts, each with one key
                    - "text": String with utterance
                - "time_num": Int
                - "time_unit": String
                - "time_back": String with a combination of 'time_unit' and 'time_back'
            - "init_personas": List with two lists with Strings
                - 0: persona sentences Speaker 1
                - 1: persona sentances Speaker 2        
        """
        all_history, all_next_utterance, all_indices = [], [], []

        if self.persona_selector is not None:
            preprocessed = {}
            if self.persona_selector.split(":")[0] == 'preprocessed':
                preprocessed = self.load_preprocessed()
            num_preprocessed = len(preprocessed.keys())

        selected_dialogues = self.dialogues
        selected_dialogue_ids = list(range(len(self.dialogues)))
        if max_samples is not None:
            if max_samples < len(selected_dialogues):
                selected_dialogue_ids = random.sample(range(len(selected_dialogues)), max_samples)
                selected_dialogues = [selected_dialogues[i] for i in selected_dialogue_ids]

        for dialog_nr, d in enumerate(selected_dialogues):
            dialog_id = selected_dialogue_ids[dialog_nr]
            turns = d.get("dialog", [])
            personas = d.get("personas", None)  # The persona sentences ('facts') that were infered from the dialogue
            init_personas = d.get("init_personas", None)  # The initial persona sentences from ConvAI2 dataset
            previous_dialogs = d.get("previous_dialogs", None)
            metadata = d.get("metadata", None)

            init_info = {"Speaker 1": [], "Speaker 2": []} # two sets, one for each speaker
            summary_info = {"Speaker 1": [], "Speaker 2": []}
            previous_utterances = []
            if self.include_persona or self.include_history:
                if previous_dialogs is not None:
                    for prev_d in previous_dialogs:
                        if self.sessionbreak_token is not None:
                            previous_utterances.append(("Nobody", prev_d["time_back"]))
                            # previous_utterances.append(("Nobody", str(prev_d['time_num']) + ' ' + prev_d['time_unit']))
                        for i in range(len(prev_d['dialog'])):
                            text = prev_d['dialog'][i].get("text", "")
                            speaker = "Speaker 1" if i % 2 == 0 else "Speaker 2"
                            previous_utterances.append((speaker, text))

            if self.include_persona:

                if self.sessionbreak_token is not None:
                    init_info = {"Speaker 1": [("Nobody", "personas")], "Speaker 2": [("Nobody", "personas")]}
                if init_personas is not None:
                    init_info["Speaker 1"] += [("Speaker 1", t) for t in init_personas[0]]
                    init_info["Speaker 2"] += [("Speaker 2", t) for t in init_personas[1]]

                if self.persona_selector is None:
                    if personas is not None:
                        # Include 'gold summary' if persona_selector not defined, corresponding to the LAST speaker
                        summary_info["Speaker 1"] = [("Speaker 1", t) for t in personas[0]]
                        summary_info["Speaker 2"] = [("Speaker 2", t) for t in personas[1]]
                else:
                    if dialog_id in preprocessed.keys():
                        summary_info = preprocessed[dialog_id]
                        logging.spam(f"Use preprocessed summary for dialog {dialog_id}")
                    else:
                        # Filter consecutive utterances from Speaker 1, Speaker 2 as summary for Speaker 2 and vice versa
                        turns_speaker1, turns_speaker2 = [], []
                        for i in range(len(previous_utterances) - 1):
                            if previous_utterances[i][0] == 'Speaker 1' and previous_utterances[i+1][0] == 'Speaker 2':
                                turns_speaker2.extend(previous_utterances[i:i+2])
                            elif previous_utterances[i][0] == 'Speaker 2' and previous_utterances[i+1][0] == 'Speaker 1':
                                turns_speaker1.extend(previous_utterances[i:i+2])
                        
                        # Extract facts from the utterances, for Speaker 1 and Speaker 2
                        summary_info["Speaker 1"] = [("Speaker 1", t) for t in self.persona_selector_fn(turns_speaker1)]
                        summary_info["Speaker 2"] = [("Speaker 2", t) for t in self.persona_selector_fn(turns_speaker2)]
                        preprocessed[dialog_id] = summary_info
                        logging.verbose(f"Extracted {len(summary_info['Speaker 1'])}+{len(summary_info['Speaker 2'])} facts from dialog {dialog_id}")

            if not self.include_history: 
                previous_utterances = []

            # Mark start of dialogue with <sessionbreak> after persona sentences, if token is provided
            start_of_session = [] if self.sessionbreak_token is None else [("Nobody", "new session")]
            current_utterances = [("Speaker 1" if i % 2 == 0 else "Speaker 2", turns[i]["text"]) for i in range(len(turns))]

            start_range = 0 if self.augmented else max(len(turns) - 1, 0)
            for len_history in range(start_range, len(turns)):
                nextspeaker = current_utterances[len_history][0]
                lastspeaker = "Speaker 1" if nextspeaker == "Speaker 2" else "Speaker 2"
                if self.input_order == 'personas-history-current':
                    history = init_info[nextspeaker] + summary_info[nextspeaker] + summary_info[lastspeaker] + previous_utterances + start_of_session + current_utterances[:len_history]
                else:
                    history = previous_utterances + init_info[nextspeaker] + summary_info[nextspeaker] + summary_info[lastspeaker] + start_of_session + current_utterances[:len_history]
                all_history.append(history)

                next_utterance = current_utterances[len_history]
                all_next_utterance.append(next_utterance)
                all_indices.append({"session": int(str(self.session)[0]), "dialog_id": dialog_id, "turn_id": len_history, "convai_id": "" if metadata is None else metadata.get("initial_data_id", "")})

        if self.persona_selector is not None:
            if len(preprocessed.keys()) > num_preprocessed:
                self.save_preprocessed(preprocessed)

        # In case of augmented dataset, check if we need to resample again
        if self.augmented and (max_samples is not None):
            if max_samples < len(all_indices):
                selected_sample_ids = random.sample(range(len(all_indices)), max_samples)
                all_indices = [all_indices[i] for i in selected_sample_ids]
                all_history = [all_history[i] for i in selected_sample_ids]
                all_next_utterance = [all_next_utterance[i] for i in selected_sample_ids]

        return all_indices, all_history, all_next_utterance
        
    def __len__(self):
        return len(self.history)
    
    def __getitem__(self, i):

        # Compose history and next utterance
        if self.speaker_prefixes is not None:
            # Determine who is Speaker 1 and Speaker 2
            mapping = self._get_speaker_mapping(i)
            # Put the right prefix before the utterance
            history = '\n'.join([self.speaker_prefixes[mapping[p]] + t for p, t in self.history[i]]) + '\n'
            next_utterance = self.speaker_prefixes["me"] + self.next_utterance[i][1] +'\n' # Also include speaker prefix for target
        else:
            history = '\n'.join([t for _, t in self.history[i]]) + '\n'
            next_utterance = self.next_utterance[i][1] +'\n'
        return history, next_utterance
    
    def _get_speaker_mapping(self, i):
        # Determine who is Speaker 1 and Speaker 2 (who is 'you', who is 'me')
        speakers = [speaker for speaker, _ in self.history[i] if speaker != "Nobody"]
        mapping = {"Speaker 1": "me", "Speaker 2": "you", "Nobody": "sessionbreak"}
        if len(speakers) > 0 and speakers[-1] == 'Speaker 1':
            mapping = {"Speaker 1": "you", "Speaker 2": "me", "Nobody": "sessionbreak"}
        return mapping

    def save_dialogue_fig(self, i, savedir='./'):

        dialog_id = self.indices[i]
        mapping = self._get_speaker_mapping(i)
        wrapped_turns = [(mapping[p], textwrap.wrap(t, width=45)) for p, t in self.history[i] + [self.next_utterance[i]]]

        variant = f"{'no' if not self.include_persona else ''}persona" + f"_{'and_' if self.include_history and self.include_persona else 'no'}history"
        title=f"Dataset: session_{self.session}/{self.subset}, dialog_id: {dialog_id['dialog_id']}\nvariant: {variant}"

        savepath = savedir + f"dialogfig_session_{self.session}_{self.subset}_{dialog_id['dialog_id']:06d}:{dialog_id['turn_id']:02d}_{variant}"
        save_dialogue_fig(wrapped_turns, title, savepath)

    def corpus(self):
        corpus = []
        for dialog in self.dialogues:
            for personas in dialog.get("personas", []):
                corpus.extend(personas)
            for utterance in dialog.get("dialog", []):
                corpus.append(utterance['text'])
        return corpus

    def item_measurements(self, i):
        stats = {
            "session": self.indices[i]["session"],
            "dialog_id": self.indices[i]["dialog_id"],
            "turn_id": self.indices[i]["turn_id"],
            "convai_id": self.indices[i]["convai_id"],
            "inputwords": len(self[i][0].split()), 
            "inputsentences": len(self.history[i]),
            "labelwords": len(self[i][1].split()), 
        }       
        return stats
    
    def measurements(self):

        num_samples = len(self)
        allitem_measurements = [self.item_measurements(i) for i in range(len(self))]
        inputwords_per_sample = Counter([m["inputwords"] for m in allitem_measurements])
        labelwords_per_sample = Counter([m["labelwords"] for m in allitem_measurements])
        totalwords_per_sample = Counter([m["inputwords"] + m["labelwords"] for m in allitem_measurements])
        inputsentences_per_sample = Counter([m["inputsentences"] for m in allitem_measurements])

        inputwords = sum([length * freq for length, freq in inputwords_per_sample.items()])
        labelwords = sum([length * freq for length, freq in labelwords_per_sample.items()])
        totalwords = sum([length * freq for length, freq in totalwords_per_sample.items()])

        all_measurements = {
            "allitem_measurements": allitem_measurements,
            "num_samples": num_samples,
            "inputwords": inputwords,
            "labelwords": labelwords,
            "totalwords": totalwords,
            "avg_inputwords": inputwords / max(num_samples, 1),
            "avg_labelwords": labelwords / max(num_samples, 1),
            "avg_totalwords": totalwords / max(num_samples, 1),
            "inputwords_per_sample": sorted(inputwords_per_sample.items(), key=lambda x:x[0]),
            "labelwords_per_sample": sorted(labelwords_per_sample.items(), key=lambda x:x[0]),
            "totalwords_per_sample": sorted(totalwords_per_sample.items(), key=lambda x:x[0]),
            "inputsentences_per_sample": sorted(inputsentences_per_sample.items(), key=lambda x:x[0])
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

                # use left padding
                # all utterances are separated by '\n', so no need to add <bos> token
                # no <eos> token at end, because end of target is marked by '\n'
                cls.tokenizer.padding_side = 'left'
                encoded = cls.tokenizer(
                    [history + next_utterance for history, next_utterance in data],
                    padding=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

            else:

                # use left padding for the input
                cls.tokenizer.padding_side = 'left'
                encoded = cls.tokenizer(
                    history_batch, 
                    padding=True, 
                    return_attention_mask=True,
                    return_tensors='pt'
                )

            # Manual truncation, necessary to calculate truncation percentage
            encoded.num_original_tokens = encoded.attention_mask.sum(dim=1)
            encoded.num_truncated_tokens = torch.zeros(len(data))
            truncated_size = encoded.input_ids.shape[1] + (buffer if not with_labels else 0) - cls.tokenizer.model_max_length
            if truncated_size > 0:
                encoded.num_truncated_tokens = encoded.attention_mask[:, :truncated_size].sum(dim=1)
                encoded["input_ids"] = encoded.input_ids[:, truncated_size:]
                encoded["attention_mask"] = encoded.attention_mask[:, truncated_size:]
            encoded.num_tokens = encoded.attention_mask.sum(dim=1)
            logging.spam(f"Encoded shape={encoded.input_ids.shape}, num_truncated_tokens={encoded.num_truncated_tokens.sum().item()}, {torch.div(encoded.num_truncated_tokens, encoded.num_original_tokens).mean().item():.2%}")

        elif batch_format == "huggingface_xysplit":

            # use left padding for the input
            cls.tokenizer.padding_side = 'left'
            encoded = cls.tokenizer(
                history_batch, 
                padding=True, 
                return_attention_mask=True,
                return_tensors='pt'
            )

            if with_labels:
                
                # use right padding for labels
                cls.tokenizer.padding_side = 'right'
                labels = cls.tokenizer(
                    next_utterance_batch,
                    padding=True, 
                    return_attention_mask=True,
                    return_tensors='pt'            
                )

            encoded.num_original_tokens = encoded.attention_mask.sum(dim=1)
            encoded.num_truncated_tokens = torch.zeros(len(data))
            truncated_size = encoded.input_ids.shape[1] + (buffer if not with_labels else labels.input_ids.shape[1]) - cls.tokenizer.model_max_length
            if truncated_size > 0:
                encoded.num_truncated_tokens = encoded.attention_mask[:, :truncated_size].sum(dim=1)
                encoded["input_ids"] = encoded.input_ids[:, truncated_size:]
                encoded["attention_mask"] = encoded.attention_mask[:, truncated_size:]
            encoded.num_tokens = encoded.attention_mask.sum(dim=1)
            logging.spam(f"Encoded shape={encoded.input_ids.shape}, num_truncated_tokens={encoded.num_truncated_tokens.sum().item()}, {torch.div(encoded.num_truncated_tokens, encoded.num_original_tokens).mean().item():.2%}")
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

        def print_responses(indices, data, responses):
            print_string = ""
            for i, (x, y), p in zip(indices, data, responses):
                print_string += f'index:      {i}\n'
                print_string += f'context:    {x}\n'
                print_string += f'target:     {y}\n'
                print_string += f'prediction: {p}\n'
                print_string += '-' * 40 + '\n'
            return print_string

        model = model.to(device)
        model.eval()
        all_responses = []
        interval_counter = 0

        # Initialize metrics
        msc_metrics = MSC_Metrics(ignore_index=self.tokenizer.pad_token_id, device=device)

        for start_index in range(0, len(self), batch_size):
            data = [self[start_index + i] for i in range(batch_size) if start_index + i < len(self)]
            indices = [self.indices[start_index + i] for i in range(batch_size) if start_index + i < len(self)]
            targets = [label for _, label in data]
            inputs, labels = self.batchify(data, with_labels=True, batch_format='huggingface_xysplit', buffer=decoder_max)
            B, L = inputs.input_ids.shape[:2]

            with torch.no_grad():
                output = model.model.generate(
                    inputs = inputs.input_ids.to(device), 
                    logits_processor=LogitsProcessorList([
                        NoRepeatNGramLogitsProcessor(ngram_size=4),
                    ]), 
                    generation_config=GenerationConfig(
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                        num_beams=1,
                        do_sample=False,
                        top_p=0.90,
                        top_k=5,
                        max_new_tokens=labels.input_ids.shape[1],
                        eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode('\n')[0]],
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                )
            responses = self.tokenizer.batch_decode(output.sequences[:, L:].to("cpu"), skip_special_tokens=True)
            all_responses.extend(responses)

            if print_max > 0:
                logging.verbose(print_responses(indices, data, responses))
                print_max -= len(data)

            msc_metrics.update(responses, targets, inputs, labels, output, indices)

            interval_counter += B
            if interval_counter >= log_interval:
                logging.verbose(f"Evaluated {len(all_responses)}/{len(self)} samples")
                interval_counter -= log_interval

        logging.info(f"Completed evaluation of {len(all_responses)} samples")

        stats, result_dict = msc_metrics.compute()

        return stats, result_dict


    @classmethod
    def predict(cls, input, model, device="cpu", decoder_max=20, batch_size=1):

        model = model.to(device)
        model.eval()
        all_responses = []

        for start_index in range(0, len(input), batch_size):
            data = [input[start_index + i] for i in range(batch_size) if start_index + i < len(input)]
            inputs = input.batchify(data, with_labels=False, batch_format='huggingface_xysplit', buffer=decoder_max)
            B, L = inputs.input_ids.shape[:2]

            with torch.no_grad():
                output = model.model.generate(
                    inputs = inputs.input_ids.to(device), 
                    logits_processor=LogitsProcessorList([
                        NoRepeatNGramLogitsProcessor(ngram_size=4),
                    ]), 
                    generation_config=GenerationConfig(
                        pad_token_id=model.model.config.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        do_sample=False,
                        top_p=0.9,
                        top_k=5,
                        max_new_tokens=decoder_max,
                        eos_token_id=[cls.tokenizer.eos_token_id, cls.tokenizer.encode('\n')[0]],
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                )
            responses = cls.tokenizer.batch_decode(output.sequences[:, L:].to("cpu"), skip_special_tokens=True)
            all_responses.extend(responses)

        return all_responses


if __name__ == "__main__":
    import argparse
    import random
    from functools import partial
    from transformers import AutoTokenizer
    from dataset.tokenizer import train_tokenizer, PAD_TOKEN
    from models.bart_extractor import BartExtractor
    from utils.general import load_config

    def get_parser():

        parser = argparse.ArgumentParser(description="Train a KnowledgeGroundedDecoder", conflict_handler='resolve')

        # General, loading, saving, logging
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--loglevel", type=str, default="SPAM", choices=logging.get_all_levels())
        parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
        
        # Training
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
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
    subset = 'valid'
    session = 4
    persona_selector = None 
    # session = "preprocessed:session_3_train_withprefixes_selectedpersona_withhistory"
    # persona_selector = 'test_bart'
    if session == 1:
        version = ['both', 'revised']
        session = '-'.join(['1'] + version)
    speaker_prefixes = ['<other>', '<self>']
    sessionbreak_token = '<sessionbreak>'
    add_tokens = None #speaker_prefixes if sessionbreak_token is None else speaker_prefixes + [sessionbreak_token]
    include_persona = True
    include_history = True
    input_order = INPUT_ORDER_OPTIONS[0]
    max_samples = 5

    augmented = False


    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if add_tokens is not None:
        tokenizer.add_tokens(add_tokens)
    tokenizer.bos_token_id = tokenizer.eos_token_id
    if speaker_prefixes is not None:
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(speaker_prefixes[0]) # Will return eos_token_id, unless <self> token had been added to tokenizer

    # tokenizer = train_tokenizer(
    #     corpus=MSC_Session(basedir=datadir + basedir, session=session, tokenizer=None, max_samples=1000).corpus(),
    #     max_size=4000
    # )
    # pad_token_id = tokenizer.token_to_id(PAD_TOKEN)

    if persona_selector is not None:

        # Load pretrained model to select generate (tokens for) persona sentences from a batch with input_ids
        loadpath = checkpoint_dir + persona_selector
        logging.info("Loading persona_selector from {}".format(loadpath))
        bart_config = load_config(loadpath + '.config')
        assert bart_config["speaker_prefixes"] == speaker_prefixes, f"persona selector was trained with speaker prefixes {bart_config['speaker_prefixes']}, current dataset has speaker prefixes {speaker_prefixes}"
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
            nofact_token=bart_config['nofact_token']
        )
        persona_selector = partial(MSC_Turns.predict_from_utterances, model=bart_model, device=bart_device, batch_size=args.batch_size)

    MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=speaker_prefixes, sessionbreak_token=sessionbreak_token)
    msc_turns = MSC_Session(
        basedir=datadir+basedir, 
        session=session, 
        subset=subset, 
        max_samples=max_samples,
        include_persona=include_persona,
        include_history=include_history,
        input_order=input_order,
        augmented=augmented,
        persona_selector=persona_selector
    )
    
    m = msc_turns.item_measurements(0)
    m = msc_turns.measurements()
    del m["allitem_measurements"]
    print(prettydict(m, title="Measurements"))

    for sentence in msc_turns.corpus()[10:30]:
        logging.verbose(sentence)
    logging.verbose('-'*40)

    for i in range(min(10, max_samples)):
        msc_turns.save_dialogue_fig(i, './output/')

    data = [msc_turns[i] for i in range(min(10, max_samples))]

    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = msc_turns.batchify(data, batch_format="huggingface_xycat")
    logging.info("Components of batch: {}".format(str(batch.keys())))
    logging.spam(batch)