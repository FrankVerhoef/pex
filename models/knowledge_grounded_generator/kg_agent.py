import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import AutoTokenizer

from models.knowledge_grounded_generator.kg_utils import NORELATION_TOKEN, ConceptGraph, blacklist
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss
from utils import logging
from utils.general import padded_tensor


class KnowledgeGroundedAgent:

    @classmethod
    def add_cmdline_args(cls, parser):
        """
        Add CLI arguments.
        """
        # Add custom arguments only for this model.
        group = parser.add_argument_group('KG Generator Agent')
        group.add_argument(
            '--kg-datadir', 
            type=str, 
            default='/users/FrankVerhoef/Programming/Project_AI/ParlAI/data/kg_data/', 
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


    def __init__(self, opt, model_tokenizer):

        self.num_hops = opt['num_hops']
        self.max_branch = opt['max_branch']
        self.max_concepts = opt['max_concepts']
        self.max_triples = opt['max_triples']
        self.overlapping_concepts = opt['overlapping_concepts']
        self.model_tokenizer = model_tokenizer
        self._cache_sorted_dict_ind = sorted(self.model_tokenizer.get_vocab().values())
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


    def observe(self, observation):
        logging.debug('=== KG AGENT - OBSERVE ===')

        text = observation['text']
        logging.debug('Text:{}'.format(text))

        labels = observation.get('labels', [])
        logging.debug("Labels: {}".format(labels))

        # Match input text and label with concepts in knowledge graph
        concepts = self.kg.match_mentioned_concepts(text, ' '.join(labels), self.overlapping_concepts)
        logging.debug("Concepts: {}:{} + {}:{}".format(
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
        #     (self.kg.id2concept[id], self.model_tokenizer.decode([self.model_tokenizer.encode(' ' + self.kg.id2concept[id])[0]]))
        #     for id in filtered_data['concept_ids']
        # ]))

        # Construct list with gate_labels
        target_concept_ids = [self.model_tokenizer.encode(' ' + c)[0] for c in concepts['target_concepts']]
        label_ids = self.model_tokenizer.encode(labels[0]) if len(labels) > 0 else []
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] #TODO: check if it is necessary to add 0 for end-token

        # Info about the related concepts
        concept_token_ids = [self.model_tokenizer.encode(' ' + self.kg.id2concept[id])[0] for id in filtered_data['concept_ids']]
        observation['concept_token_ids'] = torch.LongTensor(concept_token_ids)
        observation['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        observation['distances'] = torch.LongTensor(filtered_data['distances'])
        observation['gate_labels'] = torch.LongTensor(gate_labels)

        # Info how to map concepts to vocab
        observation['vocab_map'], observation['map_mask'] = self._build_vocab_map(concept_token_ids)

        # Info about relations to related concepts
        observation['relation_ids'] = torch.LongTensor(filtered_data['relation_ids'])
        observation['head_idx'] = torch.LongTensor(filtered_data['head_idx'])
        observation['tail_idx'] = torch.LongTensor(filtered_data['tail_idx'])
        observation['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        logging.spam("Relations {}: {}".format(
            len(observation['head_idx']),
            self.kg.formatted_triples_string(filtered_data, 5)
        ))

        return observation


    def batchify(self, obs_batch):
        batch = dict()
        batch['inputs'] = self.model_tokenizer(
            [obs['text'] for obs in obs_batch], 
            padding=True, 
            return_attention_mask=True,
            return_tensors='pt'
        )
        batch['labels'] = self.model_tokenizer(
            [obs['labels'][0] for obs in obs_batch], 
            padding=True, 
            return_attention_mask=True,
            return_tensors='pt'
        )
        batch['concept_ids'] = padded_tensor(
            [obs['concept_token_ids'] for obs in obs_batch],
            pad_value=self.model_tokenizer.pad_token_id          
        )
        batch['concept_labels'] = padded_tensor(
            [obs['concept_labels'] for obs in obs_batch],
            pad_value=-1           
        )
        batch['distances'] = padded_tensor(
            [obs['distances'] for obs in obs_batch],
            pad_value=0
        )
        batch['relation_ids'] = padded_tensor(
            [obs['relation_ids'] for obs in obs_batch], 
            pad_value=self.kg.relation2id[NORELATION_TOKEN]
        )
        batch['head_idx'] = padded_tensor(
            [obs['head_idx'] for obs in obs_batch],
            pad_value=0
        )
        batch['tail_idx'] = padded_tensor(
            [obs['tail_idx'] for obs in obs_batch],
            pad_value=0
        )
        batch['triple_labels'] = padded_tensor(
            [obs['triple_labels'] for obs in obs_batch],
            pad_value=-1
        )
        batch['gate_labels'] = padded_tensor(
            [obs['gate_labels'] for obs in obs_batch],
            pad_value=-1
        ) #TODO: should gate labels be left padded??
        batch['vocab_map'] = torch.stack(
            [obs['vocab_map'] for obs in obs_batch]
        )
        batch['map_mask'] = torch.stack(
            [obs['map_mask'] for obs in obs_batch]
        )

        return batch

    def act(self, observation):
        reply = "Let me think about that for a while"
        return(reply)
    
    def batch_act(self, obs_batch, model, device):
        batch = self.batchify(obs_batch)
        L = batch['inputs'].input_ids.shape[1]
        output = model.generate(batch, device)
        output_gen = output[:, L:]
        responses = self.model_tokenizer.batch_decode(output_gen)
        return responses

def get_parser():

    parser = argparse.ArgumentParser(description="Train a KnowledgeGroundedDecoder")

    # General, loading, saving, logging
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

    parser = get_parser()
    args = parser.parse_known_args()[0]

    parser = KnowledgeGroundedAgent.add_cmdline_args(parser)
    parser = KnowledgeGroundedDecoder.add_cmdline_args(parser)

    args = parser.parse_args()
    print(vars(args))
    logging.set_log_level(args.loglevel)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({"pad_token": '<PAD>'})
    agent = KnowledgeGroundedAgent(vars(args), tokenizer)
    model = KnowledgeGroundedDecoder(vars(args), tokenizer)
    model.to(args.device)
    criterion = KG_loss(ignore_index=tokenizer.pad_token_id, invalid=-1, alpha = args.alpha, beta = args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    obs = {
        "text": "I like my mother and sister. It is good to be with them.", #We often go out together to the park, or to a restaurant",
        "labels": ["Your family is important since birth"],
    }
    obs2 = {
        "text": "Why do you play soccer?", # I think is is a very rough game. I prefer to play tennis.",
        "labels": ["It is fun and a great sport to play as a team"],
    }
    observed = agent.observe(obs)
    observed2 = agent.observe(obs2)
    obs_batch = [observed, observed2]
    # print("OBSERVATION")
    # print(obs_batch)

    # batch = agent.batchify(obs_batch)
    # print("BATCH")
    # print(batch)

    # output = model.train_step(batch, optimizer, criterion, args.device)
    # print("OUTPUT")
    # print(output)

    responses = agent.batch_act(obs_batch, model, args.device)
    print("ACT")
    for context, response in zip(obs_batch, responses):
        print("Context:  ", context['text'])
        print("Label:    ", context['labels'])
        print("Response: ", response)
        print("-" * 20)


