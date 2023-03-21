import torch
from torch import optim
import random
from dataclasses import dataclass

from transformers import AutoTokenizer, GenerationConfig, PretrainedConfig, BatchEncoding

from models.knowledge_grounded_generator.kg_utils import NORELATION_TOKEN, ConceptGraph
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss, KG_Info
from utils import logging
from utils.general import padded_tensor

from dataset.msc_sessions import MSC_Session


class KG_enriched_MSC_Session(MSC_Session):

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
            default='./data/kg_data/', 
            help='dir for knowledge graph data'
        )
        group.add_argument(
            '--dataset-concepts', 
            type=str, 
            default='total_concepts.txt', 
            help='file with dataset concepts'
        )
        group.add_argument(
            '--kg', 
            type=str, 
            default='kg.graph', 
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
        return parser

    def __init__(self, opt, path, model_tokenizer, max_samples=None, batch_format="huggingface", batch_pad_id=0):
        super().__init__(
            path, 
            model_tokenizer, 
            speaker_prefixes=opt['speaker_prefixes'],
            include_persona=opt['include_persona'], 
            max_samples=max_samples, 
            batch_format=batch_format, 
            batch_pad_id=batch_pad_id
        )
        self.num_hops = opt['num_hops']
        self.max_branch = opt['max_branch']
        self.max_concepts = opt['max_concepts']
        self.max_triples = opt['max_triples']
        self.overlapping_concepts = opt['overlapping_concepts']
        self._cache_sorted_dict_ind = sorted(self.tokenizer.get_vocab().values())
        self.kg = ConceptGraph(opt['kg_datadir'], opt['kg'])
        self.kg.build_reduced_graph(opt['kg_datadir'] + opt['dataset_concepts'])
        logging.info("Initialized KnowledgeGroundedAgent")


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
        logging.verbose('Get KG info')

        kg_info = dict()
        if labels is None:
            labels = []
        logging.verbose("Text  : {}".format(text))
        logging.verbose("Labels: {}".format(labels))

        # Match input text and label with concepts in knowledge graph
        concepts = self.kg.match_mentioned_concepts(text, ' '.join(labels), self.overlapping_concepts)
        logging.verbose("Concepts: {}:{} + {}:{}".format(
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

        logging.verbose("Related concepts {}: {}".format(
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
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] #TODO: check if it is necessary to add 0 for end-token

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

        logging.verbose("Relations {}: {}".format(
            len(kg_info['head_idx']),
            self.kg.formatted_triples_string(filtered_data, 5)
        ))

        return kg_info


    def batchify(self, data):

        # seperate source and target sequences and kg_info
        text_batch, labels_batch, kg_info_batch = zip(*data)

        inputs = self.tokenizer(
            [text for text in text_batch], 
            padding=True, 
            return_attention_mask=True,
            return_tensors='pt'
        )

        # tokenizer uses LEFT padding, but need to use RIGHT padding for labels
        tokenized_labels = [self.tokenizer.encode(labels[0], return_tensors='pt').squeeze() for labels in labels_batch]
        labels = BatchEncoding({
            'input_ids': padded_tensor(tokenized_labels, pad_value=self.tokenizer.pad_token_id),
            'attention_mask': padded_tensor([torch.ones(len(sequence), dtype=torch.long) for sequence in tokenized_labels], pad_value=0)
        })

        kg_info = KG_Info(
            concept_ids = padded_tensor([obs['concept_token_ids'] for obs in kg_info_batch], pad_value=self.tokenizer.pad_token_id),
            concept_labels = padded_tensor([obs['concept_labels'] for obs in kg_info_batch], pad_value=-1),
            distances = padded_tensor([obs['distances'] for obs in kg_info_batch], pad_value=0),
            relation_ids = padded_tensor([obs['relation_ids'] for obs in kg_info_batch], pad_value=self.kg.relation2id[NORELATION_TOKEN]),
            head_idx = padded_tensor([obs['head_idx'] for obs in kg_info_batch], pad_value=0),
            tail_idx = padded_tensor([obs['tail_idx'] for obs in kg_info_batch], pad_value=0),
            triple_labels = padded_tensor([obs['triple_labels'] for obs in kg_info_batch], pad_value=-1),
            gate_labels = padded_tensor([obs['gate_labels'] for obs in kg_info_batch], pad_value=-1), #TODO: should gate labels be left padded??
            vocab_map = torch.stack([obs['vocab_map'] for obs in kg_info_batch]),
            map_mask = torch.stack([obs['map_mask'] for obs in kg_info_batch])
        )

        return inputs, labels, kg_info

    
def batch_act(obs_batch, model, tokenizer, device, collate_fn):
    inputs, labels, kg_info = collate_fn(obs_batch)
    L = inputs.input_ids.shape[1]
    model.to(device)
    input_ids = inputs.input_ids.to(device)
    kg_input = kg_info.to(device)
    output = model.generate(
        inputs=input_ids,
        kg_input=kg_input,
        generation_config=GenerationConfig(
            pad_token_id=model.gpt2model.config.eos_token_id,
            use_cache=True,
            num_beams=1,
            do_sample=False,
            max_new_tokens=20
        )
    )
    output_gen = output[:, L:]
    responses = tokenizer.batch_decode(output_gen)
    return responses

def get_parser():

    parser = argparse.ArgumentParser(description="Train a KnowledgeGroundedDecoder")

    # General, loading, saving, logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--loglevel", type=str, default="DEBUG", choices=logging.get_all_levels())    
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    
    # Dataset
    parser.add_argument("--datadir", type=str, default="/Users/FrankVerhoef/Programming/PEX/data/", help="Datadir")
    parser.add_argument("--traindata", type=str, default="msc/msc_personasummary/session_1/train.txt", help="Dataset file for training")
    parser.add_argument("--validdata", type=str, default="msc/msc_personasummary/session_1/valid.txt", help="Dataset file for validation")
    parser.add_argument("--testdata", type=str, default="msc/msc_personasummary/session_1/test.txt", help="Dataset file for testing")
    parser.add_argument("--train_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--valid_samples", type=int, default=None, help="Max number of test samples")
    parser.add_argument("--test_samples", type=int, default=None, help="Max number of test samples")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    return parser

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = get_parser()
    args = parser.parse_known_args()[0]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    parser = KG_enriched_MSC_Session.add_cmdline_args(parser)
    parser = KnowledgeGroundedDecoder.add_cmdline_args(parser)

    args = parser.parse_args()
    print(vars(args))
    logging.set_log_level(args.loglevel)
    logging.info("Unit test {}".format(__file__))

    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_dialogue/session_2/train.txt'
    dataset = KG_enriched_MSC_Session(
        vars(args), 
        datapath, 
        tokenizer, 
        max_samples=None, 
        batch_format="huggingface", 
        batch_pad_id=tokenizer.pad_token_id
    )
    model = KnowledgeGroundedDecoder(vars(args), tokenizer, config=PretrainedConfig())
    criterion = KG_loss(ignore_index=tokenizer.pad_token_id, invalid=-1, alpha = args.alpha, beta = args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Test extraction of dialogue turns and persona sentences
    # msc_turns = MSC_Session(datapath, tokenizer, speaker_prefixes=['<self>', '<other>'], include_persona=True, batch_format="huggingface", batch_pad_id=-1)
    # data = [msc_turns[i] for i in range(10)]

    data = [dataset[i] for i in range(5)]
    for item in data:
        logging.verbose(item[0])
        logging.verbose(item[1])
        logging.verbose('-'*40)

    batch = dataset.batchify(data)
    logging.spam("BATCH\n{}".format(batch))

    output = model.train_step(batch, optimizer, criterion, args.device)
    logging.report("Train_step output: {}".format(output))

    output = model.valid_step(batch, criterion, args.device)
    logging.report("Valid_step output: {}".format(output))

    responses = batch_act(data, model, tokenizer, device=args.device, collate_fn=dataset.batchify)
    print("GENERATE")
    for (text, label, kg_info), response in zip(data, responses):
        print("Context:  ", text)
        print("Label:    ", label)
        print("Response: ", response)
        print("-" * 20)

