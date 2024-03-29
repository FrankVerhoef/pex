import torch
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
import evaluate

from transformers import BatchEncoding, GenerationConfig

from models.knowledge_grounded_generator.kg_utils import NORELATION_TOKEN, ConceptGraph
from models.knowledge_grounded_generator.kg_model import KG_Info
from utils import logging
from utils.general import padded_tensor

from dataset.msc_sessions import MSC_Session
from dataset.convai2 import ConvAI2


class KG_enriched_MSC_Session(MSC_Session):

    @classmethod
    def set(cls, tokenizer, speaker_prefixes, kg, num_hops, max_branch, max_concepts, max_triples, overlapping_concepts):
        super().set(tokenizer, speaker_prefixes)
        cls._cache_sorted_dict_ind = sorted(cls.tokenizer.get_vocab().values())
        cls.kg = kg
        cls.num_hops = num_hops
        cls.max_branch = max_branch
        cls.max_concepts = max_concepts
        cls.max_triples = max_triples
        cls.overlapping_concepts = overlapping_concepts

    @classmethod
    def add_cmdline_args(cls, parser):
        """
        Add CLI arguments.
        """
        # Add custom arguments only for this model.
        parser = super().add_cmdline_args(parser)
        group = parser.add_argument_group('KG enriched MSC dataset')
        group.add_argument(
            '--kg-datadir', 
            type=str, 
            default='/users/FrankVerhoef/Programming/PEX/data/kg_data/', 
            help='dir for knowledge graph data'
        )
        group.add_argument(
            '--dataset-concepts', 
            type=str, 
            default='dataset_concepts.txt', 
            help='file with dataset concepts'
        )
        group.add_argument(
            '--kg', 
            type=str, 
            default='kg.graph-sm', 
            help='file with knowledge graph'
        )
        group.add_argument(
            "--max-concepts",
            type=int,
            default=256,
            help="Maximum number of related concepts to include."
        )
        group.add_argument(
            "--max-triples",
            type=int,
            default=768,
            help="Maximum number of relations to include."
        )
        group.add_argument(
            "--max-branch",
            type=int,
            default=64,
            help="Maximum number of related concepts to add per hop."
        )
        group.add_argument(
            "--overlapping-concepts",
            type=str,
            choices=["excl-tgt-in-src", "excl-src-in-tgt", "keep-src-and-tgt"],
            default="excl-src-in-tgt",
            help="How to ensure disjoint sets of concepts."
        )
        group.add_argument(
            "--num_hops",
            type=int,
            default=2,
            help="Number of hops in the graph to look for related concepts."
        )
        return parser

    def __init__(self, 
        basedir='./', 
        session=2,
        subset='train', 
        include_persona=False,
        max_samples=None
    ):
        super().__init__(
            basedir=basedir,
            session=session, 
            subset=subset,
            include_persona=include_persona, 
            max_samples=max_samples, 
        )
        logging.info("Initialized KG_enriched_MSC_Session")


    def _build_vocab_map(self, concept_token_ids):
        """
        The vocab map and associated mask are a mapping between the GPT2 vocabulary and the KG concepts
        in the current observation.
        At each position in the vocab map, the value is the index in the list of concepts that are 
        present in the observation. The vocab mask and map mask are used in the KGG-model to map the 
        calculated concept-scores back to token-scores in the GPT2 vocabulary.
        """
        vocab_map = torch.zeros(len(self._cache_sorted_dict_ind), dtype=torch.long)
        map_mask = torch.zeros_like(vocab_map)
        for i, token_id in enumerate(self._cache_sorted_dict_ind):
            try: 
                pos = concept_token_ids.index(token_id)
                vocab_map[i] = pos
                map_mask[i] = 1
            except ValueError:
                pass

        return vocab_map, map_mask

    def __getitem__(self, i):
        text, label = super().__getitem__(i)
        kg_info = self._get_kg_info(text, [label])
        # kg_info.drop('text')
        # kg_info.drop('labels')
        return text, [label], kg_info

    def _get_kg_info(self, text, labels=None):
        logging.spam('Get KG info')

        kg_info = dict()
        if labels is None:
            labels = []
        logging.spam("Text  : {}".format(text))
        logging.spam("Labels: {}".format(labels))

        # Match input text and label with concepts in knowledge graph
        concepts = self.kg.match_mentioned_concepts(text, ' '.join(labels), self.overlapping_concepts)
        logging.spam("Concepts: {}:{} + {}:{}".format(
            len(concepts['source_concepts']), 
            concepts['source_concepts'], 
            len(concepts['target_concepts']), 
            concepts['target_concepts'])
        )
        if len(concepts['source_concepts']) > self.max_concepts:
            logging.warning("Number of source concepts {} is larger than max-concepts {}. If this happens frequently, consider to increase max-concepts".format(
                len(concepts['source_concepts']), self.max_concepts)
            )

        # Find related concepts and connecting triples
        related_concepts = self.kg.find_neighbours_nx(
            concepts['source_concepts'], 
            concepts['target_concepts'], 
            num_hops=self.num_hops, 
            max_B=self.max_branch
        )
        num_concepts = len(related_concepts['concept_ids'])
        num_triples = len(related_concepts['triples'])
        filtered_data = self.kg.filter_directed_triple(related_concepts, max_concepts=self.max_concepts, max_triples=self.max_triples)
        del related_concepts

        logging.spam("Related concepts {}: {}".format(
            len(filtered_data['concept_ids']), 
            self.kg.formatted_concepts_string(filtered_data, 10)
        ))
        # logging.spam("Translated concepts: {}".format([
        #     (self.kg.id2concept[id], self.tokenizer.decode([self.tokenizer.encode(' ' + self.kg.id2concept[id])[0]]))
        #     for id in filtered_data['concept_ids']
        # ]))

        # Construct list with gate_labels
        target_concept_ids = [self.tokenizer.encode(' ' + c)[0] for c in concepts['target_concepts']]
        label_ids = self.tokenizer.encode(labels[0]) if len(labels) > 0 else []
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] + [0] #Added 0, because 'eos_token' will be added to labels

        # Info about the related concepts
        concept_token_ids = [self.tokenizer.encode(' ' + self.kg.id2concept[id])[0] for id in filtered_data['concept_ids']]
        kg_info['concept_token_ids'] = torch.LongTensor(concept_token_ids)
        kg_info['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        kg_info['distances'] = torch.LongTensor(filtered_data['distances'])
        kg_info['gate_labels'] = torch.LongTensor(gate_labels)

        # Info how to map concepts to vocab
        kg_info['vocab_map'], kg_info['map_mask'] = self._build_vocab_map(concept_token_ids)

        # Info about relations to related concepts
        kg_info['relation_ids'] = torch.LongTensor(filtered_data['relation_ids'])
        kg_info['head_idx'] = torch.LongTensor(filtered_data['head_idx'])
        kg_info['tail_idx'] = torch.LongTensor(filtered_data['tail_idx'])
        kg_info['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        logging.spam("Relations {}: {}".format(
            len(kg_info['head_idx']),
            self.kg.formatted_triples_string(filtered_data, 5)
        ))

        return kg_info

    @classmethod
    def batchify(cls, data, batch_format):
        # TODO: make sure total length of input + labels fits in transformer!

        assert batch_format == "huggingface_kg", f"batch_format '{batch_format}' not supported by {cls.__name__}"

        # seperate source and target sequences and kg_info
        text_batch, labels_batch, kg_info_batch = zip(*data)

        # use left padding for the input
        cls.tokenizer.padding_size = 'left'
        inputs = cls.tokenizer(
            text_batch, 
            padding=True, 
            max_length=cls.tokenizer.model_max_length, 
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # use right padding for labels
        # add 'eos_token' at end of all labels (necessary to make sure 'shift_right' of labels is possible )
        cls.tokenizer.padding_side = 'right'
        labels = cls.tokenizer(
            [labels[0] + cls.tokenizer.eos_token for labels in labels_batch],
            padding=True, 
            max_length=cls.tokenizer.model_max_length, 
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'            
        )
        total_size = inputs.input_ids.shape[1] + labels.input_ids.shape[1]
        if total_size > cls.tokenizer.model_max_length:
            truncated_size = cls.tokenizer.model_max_length - labels.input_ids.shape[1]
            inputs.input_ids = inputs.input_ids[:, -truncated_size:]
            inputs.attention_mask = inputs.attention_mask[:, -truncated_size:]

        kg_info = KG_Info(
            concept_ids = padded_tensor([obs['concept_token_ids'] for obs in kg_info_batch], pad_value=cls.tokenizer.pad_token_id),
            concept_labels = padded_tensor([obs['concept_labels'] for obs in kg_info_batch], pad_value=-1),
            distances = padded_tensor([obs['distances'] for obs in kg_info_batch], pad_value=0),
            relation_ids = padded_tensor([obs['relation_ids'] for obs in kg_info_batch], pad_value=cls.kg.relation2id[NORELATION_TOKEN]),
            head_idx = padded_tensor([obs['head_idx'] for obs in kg_info_batch], pad_value=0),
            tail_idx = padded_tensor([obs['tail_idx'] for obs in kg_info_batch], pad_value=0),
            triple_labels = padded_tensor([obs['triple_labels'] for obs in kg_info_batch], pad_value=-1),
            gate_labels = padded_tensor([obs['gate_labels'] for obs in kg_info_batch], pad_value=-1),
            vocab_map = torch.stack([obs['vocab_map'] for obs in kg_info_batch]),
            map_mask = torch.stack([obs['map_mask'] for obs in kg_info_batch])
        )

        return inputs, labels, kg_info


    def evaluate(self, model, device="cpu", decoder_max=20, batch_size=1, print_max=20, log_interval=100):

        def print_responses(data, responses):
            for (x, y, _), p in zip(data, responses):
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
            inputs, _, kg_input = self.batchify(data, model.batch_format)
            B, L = inputs.input_ids.shape[:2]
            bos_tokens = torch.full((B, 1), fill_value=model.bos_token_id, dtype=torch.long, device=inputs.input_ids.device)

            with torch.no_grad():
                output = model.generate(
                    # Add the bos_token to the input. 
                    inputs=torch.cat([inputs.input_ids, bos_tokens], dim=1).to(device), 
                    kg_input=kg_input.to(device),
                    generation_config=GenerationConfig(
                        pad_token_id=model.gpt2model.config.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=decoder_max
                    )
                )
                output = output.cpu()
            responses = self.tokenizer.batch_decode(output[:, L+1:])

            if print_max > 0:
                print_responses(data, responses)
                print_max -= len(data)

            target_responses.extend([labels[0] for _, labels, _ in data])
            pred_responses.extend(responses)

            interval_counter += len(pred_responses)
            if interval_counter >= log_interval:
                logging.verbose(f"Evaluated {len(pred_responses)}/{self.__len__()} samples")
                interval_counter -= log_interval

        logging.info(f"Completed evaluation of {len(pred_responses)} samples")

        bleu_google = google_bleu.compute(predictions=pred_responses, references=[[t] for t in target_responses])
        meteor_score = meteor.compute(predictions=pred_responses, references=[[t] for t in target_responses])
        bleu_2 = bleu_score(pred_responses, target_responses, n_gram=2, smooth=True).item()
        bleu_4 = bleu_score(pred_responses, target_responses, n_gram=4, smooth=True).item()
        rouge_scores = rouge_score(pred_responses, target_responses, rouge_keys=('rouge1', 'rouge2', 'rougeL'))

        stats = {
            "bleu_2": bleu_2, 
            "bleu_4": bleu_4, 
            "gleu": bleu_google, 
            "meteor": meteor_score
        }
        stats.update({k: v.item() for k, v in rouge_scores.items()})

        return stats, {} # need to add result_dict


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test KG_enriched_MSC_Session", conflict_handler="resolve")
    parser = KG_enriched_MSC_Session.add_cmdline_args(parser)
    parser = ConvAI2.add_cmdline_args(parser)

    args = parser.parse_args()
    print(vars(args))
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    datadir = '/Users/FrankVerhoef/Programming/PEX/data/'
    basedir = 'msc/msc_dialogue/'
    subset = 'train'
    args.session = 2
    if args.session == 1:
        version = args.convai2_version
        args.session = '-'.join(['1'] + version)

    kg = ConceptGraph(args.kg_datadir, args.kg)
    kg.build_reduced_graph(args.kg_datadir + args.dataset_concepts)

    dataset_config = vars(args)
    dataset_config = {
        'basedir': datadir + basedir,
        'session': args.session,
        'max_samples': None
    }

    KG_enriched_MSC_Session.set(
        tokenizer, args.speaker_prefixes, 
        kg, args.num_hops, args.max_branch, args.max_concepts, args.max_triples, args.overlapping_concepts
    )
    dataset = KG_enriched_MSC_Session(
        **dataset_config
    )

    # Test extraction of dialogue turns and persona sentences
    # msc_turns = MSC_Session(
    #     basedir=datadir+basedir, 
    #     session=args.session, 
    #     subset=subset, 
    #     tokenizer=tokenizer, 
    #     speaker_prefixes=['<self>', '<other>'], 
    #     include_persona=True, 
    #     batch_format="huggingface", 
    #     batch_pad_id=-1
    # )
    # data = [msc_turns[i] for i in range(10)]

    data = [dataset[i] for i in range(5)]
    itemstrings = [
        "Text:   {}\n" \
        "Labels: {}\n" \
        "{}\n".format(item[0], item[1], '-'*40)
        for item in data
    ]
    logging.verbose("ITEMS\n{}".format('\n'.join(itemstrings)))

    batch = dataset.batchify(data, batch_format="huggingface_kg")
    logging.spam("BATCH\n{}".format(batch))
