import networkx as nx
import json
import pickle
import spacy
import csv
from utils import logging

HEAD=0
REL=1
TAIL=2

nlp = spacy.load('en_core_web_sm', disable=['parser', 'lemmatizer'])

# Blacklist contains concepts NOT to include.
# In this case, it is the list with auxiliary verbs and the word 'persona' 
# which is used as identifier in the input text for the persona descriptions.
blacklist = set([
    "persona", 
    "be", "am", "is", "are", "was", "were", "being", "been",
    "do", "does", "did",
    "can", "could",
    "will", "would", "wo",
    "have", "has", "had",
    "must",
    "shall", "should",
    "may", 
    "might",
    "dare", 
    "need",
    "ought"
])


NOCONCEPT_TOKEN = '<NoConcept>'
NORELATION_TOKEN = '<NoRelation>'

class ConceptGraph(nx.Graph):

    def __init__(self, path, graph):
        super().__init__()
        self.load_knowledge_graph(path + graph)


    def load_knowledge_graph(self, graph_path):
        """
            Load the graph and store it
        """
        kg = pickle.load(open(graph_path, "rb"))
        self.graph = kg["graph"]
        self.id2concept = kg["concepts"]
        self.id2concept.append(NOCONCEPT_TOKEN)        # can be used as padding
        self.concept2id = {cpt: i for i, cpt in enumerate(self.id2concept)}

        self.id2relation = kg["relations"]
        self.id2relation.extend(['*' + rel for rel in kg["relations"]])
        self.id2relation.append(NORELATION_TOKEN)      # can be used as padding
        self.relation2id = {rel: i for i, rel in enumerate(self.id2relation)}

        logging.info("Loaded knowledge graph with {} concepts, {} relation types and {} edges".format(
            len(self.graph.nodes), len(self.id2relation)-1, len(self.graph.edges)
        ))


    def build_reduced_graph(self, concepts_path):
        """
            Construct a subgraph with only the nodes in the given concepts_subset.
            In this graph multiple edges between the same nodes are combined (and weight added)
        """
        # dataset_concepts is the set of concepts that appears in the dataset (train and validation dialogues)
        with open(concepts_path, 'r') as f:
            dataset_concepts = set([line[:-1] for line in f])

        # concepts_subset is the dataset concepts, minus words on the blacklist (e.g. auxiliary verbs) 
        # that also appear in the knowledge graph
        concepts_subset = (dataset_concepts - blacklist).intersection(self.id2concept)
        concepts_subset_ids = [self.concept2id[c] for c in concepts_subset if c in self.concept2id.keys()]

        # Create subgraph with only concepts that apprar in the dataset
        cpnet_simple = nx.Graph()
        for u, v, data in nx.subgraph(self.graph, concepts_subset_ids).edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)

        self.simple_graph = cpnet_simple
        self.simple_vocab = [self.id2concept[u] for u in cpnet_simple.nodes()]

        logging.info("Built reduced graph with {} nodes and {} edges".format(len(cpnet_simple.nodes), len(cpnet_simple.edges)))


    def hard_ground(self, sent):
        """
            Returns a list of verbs and nouns in the input sentence
            that also occur in the (reduced) ConceptNet vocabulary
        """
        # TODO: Need to think about how to match words with concepts in ConceptNet
        # 'father in law' is 1 concept in ConceptNet, but Spacy tokenizer breaks it into three words,
        # 'father', 'in', 'law' which also occur in ConceptNet

        sent = sent.lower()
        doc = nlp(sent)
        result = set()

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                if token.text in self.simple_vocab:
                    result.add(token.text)

        return result


    def match_mentioned_concepts(self, source, target, overlapping_concepts):
        """
            Returns a dict with concepts from the input sentence and the target sentence
        """

        if overlapping_concepts == "keep-src-and-tgt":
            source_concepts = self.hard_ground(source)
            target_concepts = self.hard_ground(target)            
        else:
            combined_concepts = self.hard_ground(source + ' ' + target)
            if overlapping_concepts == "excl-tgt-in-src":
                target_concepts = self.hard_ground(target)
                source_concepts = combined_concepts - target_concepts
            else: # excl-src-in-tgt
                source_concepts = self.hard_ground(source)
                target_concepts = combined_concepts - source_concepts
        return {"source_concepts": list(source_concepts), "target_concepts": list(target_concepts)}


    def get_relations(self, src_concept, tgt_concept):
            
        try:
            rel_list = self.graph[src_concept][tgt_concept]
            return list(set([rel_list[item]["rel"] for item in rel_list]))
        except:
            return []


    def find_neighbours_nx(self, source_concepts, target_concepts, num_hops, max_B=100):
        """
            Find neighboring concepts within num_hops and the connecting triples
            Use graph functions from networkx
        """
        # id's in knowledge graph of the source and target concepts   
        source_ids = set(self.concept2id[s_cpt] for s_cpt in source_concepts)
        target_ids = [self.concept2id[t_cpt] for t_cpt in target_concepts]
        all_concepts = self.simple_graph.nodes

        # Vts init contains id's of source concepts with distance 0
        Vts = dict([(x,0) for x in source_ids])
        Ets = {}

        related = source_ids
        current_boundary = source_ids
        for t in range(num_hops):
            V = {}
            for v in nx.node_boundary(self.simple_graph, current_boundary, all_concepts - related):
                incoming_nodes = list(nx.node_boundary(self.simple_graph, [v], current_boundary))
                V[v] = sum(
                    self.simple_graph[u][v].get('weight', 1) 
                    for u in incoming_nodes
                )
                Ets.update(dict([(v, dict([
                    (u, self.get_relations(u, v))
                    for u in incoming_nodes
                ]))]))

            # Select nodes that are 'most' connected
            top_V = sorted(list(V.items()), key=lambda x: x[1], reverse=True)[:max_B]
            new_boundary = [v[0] for v in top_V]

            # Add nodes to Vts, with distance increased by 1
            Vts.update(dict([(v, t+1) for v in new_boundary]))
            related.update(new_boundary)
            current_boundary = new_boundary

        concept_ids = [id for id in Vts.keys()]
        labels = [int(c in target_ids) for c in concept_ids]
        distances = [d for d in Vts.values()]
        triples = [
            (u, rels, v)
            for v, incoming_relations in Ets.items()
            for u, rels in incoming_relations.items()
            if (u in concept_ids) and (v in concept_ids)
        ]

        return {"concept_ids":concept_ids, "labels":labels, "distances":distances, "triples":triples}


    def filter_directed_triple(self, related_concepts, max_concepts=64, max_triples=256):

        num_concepts = len(related_concepts['concept_ids'])
        if num_concepts > max_concepts:
            num_concepts = max_concepts
        concept_ids = related_concepts['concept_ids'][:max_concepts]
        labels = related_concepts['labels'][:max_concepts]
        distances = related_concepts['distances'][:max_concepts]
        triples = related_concepts['triples']

        # Construct triple_dict, with per tail-node, all the triples that are connected
        triple_dict = {}
        for triple in triples:
            head, _, tail = triple
            try:
                head_index = concept_ids.index(head)
                tail_index = concept_ids.index(tail)
                if distances[head_index] <= distances[tail_index]:
                    if tail not in triple_dict:
                        triple_dict[tail] = [triple]
                    else:
                        triple_dict[tail].append(triple)
            except ValueError:
                # If head or tail not found in concept_ids (because of truncation), just pass
                pass

        targets = [id for id, l in zip(concept_ids, labels) if l == 1]
        sources = [id for id, d in zip(concept_ids, distances) if d == 0]
        shortest_paths = []
        for target in targets:
            shortest_paths.extend(bfs(target, triple_dict, sources))

        ground_truth_triples_set = set([
            (n, path[i+1]) 
            for path in shortest_paths 
            for i, n in enumerate(path[:-1])
        ])

        heads, tails, relations, triple_labels = [], [], [], []
        triple_count = 0

        # Sort triple lists, one list per tail node
        triple_lists_sorted = sorted(list(triple_dict.values()), key=lambda x: len(x), reverse=False)
        num_triples = sum([len(triple_list) for triple_list in triple_lists_sorted])
        if num_triples > max_triples:
            num_triples = max_triples

        # Loop through triple lists. This can never be more than max_triples; rest is truncated
        num_triple_lists = min(max_triples, len(triple_lists_sorted))
        for i, triple_list in enumerate(triple_lists_sorted[:num_triple_lists]):
            max_neighbors = (max_triples - triple_count) // (num_triple_lists - i)
            for (head, rels, tail) in triple_list[:max_neighbors]:
                heads.append(concept_ids.index(head))
                tails.append(concept_ids.index(tail))
                relations.append(rels[0])   # Keep only one relation
                triple_labels.append(int((tail, head) in ground_truth_triples_set))
                triple_count += 1

        logging.debug("Connecting paths: {} with {} triples; kept {} triples with {} targets".format(
            len(shortest_paths), len(ground_truth_triples_set), len(triple_labels), sum(triple_labels)
        ))
        if len(shortest_paths) > 0:
            logging.debug("Examples: {}".format([
                " - ".join([self.id2concept[n] for n in p])
                for p in shortest_paths
            ][:5]))

        filtered_concepts = {
            "concept_ids": concept_ids,
            "labels": labels,
            "distances": distances,
            "head_idx": heads,
            "tail_idx": tails,
            "relation_ids": relations,
            "triple_labels": triple_labels
        }
            
        return filtered_concepts


    def formatted_concepts_string(self, related_concepts, max):
        concepts = [self.id2concept[id] for id in related_concepts['concept_ids']]
        if len(concepts) > max and max > 4:
            return 'Examples: ' + ', '.join(concepts[:max//2]) + ' ... ' + ', '.join(concepts[-max//2:])
        else:
            return(', '.join(concepts))
 

    def formatted_triples_string(self, related_concepts, max):
        n = max if len(related_concepts['relation_ids']) > max else len(related_concepts['relation_ids'])
        return ', '.join([
            '({}, {}, {}) '.format(
                self.id2concept[related_concepts['concept_ids'][related_concepts['head_idx'][i]]],
                self.id2relation[related_concepts['relation_ids'][i]],
                self.id2concept[related_concepts['concept_ids'][related_concepts['tail_idx'][i]]]
            )
            for i in range(n)
        ])


def bfs(target, triple_dict, sources, max_steps=2):
    """
        Perform breath-first-search and return all paths that connect nodes in sources to target
    """
    paths = [[[target]]]
    connecting_paths = []
    for _ in range(max_steps):
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            for triple in triple_dict.get(path[-1], []):
                new_paths.append(path + [triple[HEAD]])

        for path in new_paths:
            if path[-1] in sources:
                connecting_paths.append(path)
        
        paths.append(new_paths)
    
    return connecting_paths


import torch
def last_token_info(model_output, decoder, print_max=8):

    B = model_output.logits.shape[0]
    if B < print_max:
        print_max = B
    m = 7
    top5probs = torch.topk(model_output.logits[:print_max, -1, :], k=5, dim=-1, sorted=True)
    top5lm = torch.topk(model_output.lm_probs[:print_max, -1, :], k=5, dim=-1, sorted=True)
    top5kg = torch.topk(model_output.concept_probs_vocab[:print_max, -1, :], k=5, dim=-1, sorted=True)
    gate = model_output.gate[:print_max, -1]
    is_concept = model_output.is_concept[:print_max, -1]
    print_str = '--- token probabilities ---\n'
    print_str += " ".join([
        "{:.2f} {:7s}".format(values[0].item(), decoder(indices[0].item())[:m]) 
        for values, indices in zip(list(top5probs[0]), list(top5probs[1]))
    ]) +'\n'
    print_str += " ".join([
        "     {:7s}".format('C' if c else ' ')
        for c in is_concept
    ]) + '\n'
    print_str += " ".join([
        "{:.2f}        ".format(v.item())
        for v in gate
    ]) + '\n'
    print_str += "\n"
    print_str += '\n'.join([
        " ".join([
            "{:.2f} {:7s}".format(top5lm[0][b][index].item(), decoder(top5lm[1][b][index].item())[:m]) 
            for b in range(print_max)
        ])
        for index in range(5)
    ]) + '\n'
    print_str += "-" * 104 + '\n'
    print_str += '\n'.join([
        " ".join([
            "{:.2f} {:7s}".format(top5kg[0][b][index].item(), decoder(top5kg[1][b][index].item())[:m]) 
            for b in range(print_max)
        ])
        for index in range(5)
    ]) + '\n'
    print_str += "-" * 104 + '\n'
    return print_str