import torch
from torch import masked_fill
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_max, scatter_mean, scatter_add
from transformers import GPT2LMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from utils import logging


class KG_loss(nn.Module):

    def __init__(self, ignore_index, invalid, alpha, beta):
        super().__init__()
        self.ignore_index = ignore_index
        self.invalid = invalid
        self.alpha = alpha
        self.beta = beta
        self.gen_loss_fn = nn.NLLLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, lm_probs, labels, triple_prob, triple_labels, gate, gate_labels):
        B = lm_probs.size(0)

        # Compute generation loss
        num_target_tokens = labels.ne(self.ignore_index).long().sum(dim=-1)
        probs_clamp = lm_probs.clamp(min=1e-5)
        gen_loss_token = self.gen_loss_fn(probs_clamp.log().view(-1, lm_probs.size(-1)), labels.view(-1)).view(B, -1)
        gen_loss = gen_loss_token.sum(dim=-1) / num_target_tokens.clamp(min=1)

        # Compute triple loss
        triple_mask = (triple_labels != self.invalid).unsqueeze(1).expand_as(triple_prob).float()
        num_valid_triples = triple_mask.sum(dim=(-2,-1))
        triple_labels = triple_labels.unsqueeze(1).expand_as(triple_prob) * triple_mask
        triple_loss_fn = nn.BCELoss(weight=triple_mask, reduction='none')
        triple_loss_triple = triple_loss_fn(triple_prob, triple_labels.float()).view(B, -1)
        triple_loss = triple_loss_triple.sum(dim=-1) / num_valid_triples.clamp(min=1)

        # Compute gate loss   
        gate_mask = (gate_labels != self.invalid).float()
        gate_labels.masked_fill_((gate_labels == self.invalid), 0)
        lm_mask = (gate_labels.sum(1) != 0).float().unsqueeze(1)
        gate_mask = lm_mask.expand_as(gate_labels) * gate_mask
        num_valid_gates = gate_mask.sum(dim=-1)
        gate_loss_fn = nn.BCELoss(weight=gate_mask, reduction='none')
        gate_loss_token = gate_loss_fn(gate.view(B, -1), gate_labels.float()).view(B, -1)
        gate_loss = gate_loss_token.sum(dim=-1) / num_valid_gates.clamp(min=1)

        combined_loss = gen_loss + self.alpha * gate_loss + self.beta * triple_loss

        return combined_loss, gen_loss, triple_loss, gate_loss   


class TripleEncoder(nn.Module):

    def __init__(self, embedding, num_hops):
        super().__init__()

        E = embedding.weight.shape[-1]
        self.num_hops = num_hops
        self.concept_embd = embedding
        self.relation_embd = nn.Embedding(69, E) # 2x34+1, for 2 x #relation types in ConceptNet5 + NORELATION token
        self.W_s = nn.ModuleList([nn.Linear(E, E, bias=False) for _ in range(self.num_hops)]) 
        self.W_n = nn.ModuleList([nn.Linear(E, E, bias=False) for _ in range(self.num_hops)]) 
        self.W_r = nn.ModuleList([nn.Linear(E, E, bias=False) for _ in range(self.num_hops)])
        self.act = nn.ReLU()

        logging.info("Initialized TripleEncoder")


    def forward(self, kg_input):
        """
        Encodes knowledge triples
        Tensor sizes are:
            B, Mc: for concept_ids
            B, Mt: for relations, head_ids, tail_ids
            B, Mt, 3 x E: for output (triple representation)

            B = batch size
            Mc = number of related concepts (can vary per batch)
            Mt = number of related triples (can vary per batch)
            E = embedding dimension for concepts (is same as embedding dim for relations)
        """
        logging.debug("Forward TripleEncoder")
        logging.debug("\tEncoding {} concepts and {} relations".format(kg_input['concept_ids'].shape, kg_input['relation_ids'].shape))

        # Embed concepts and relations
        concept_repr = self.concept_embd(kg_input['concept_ids'])
        rel_repr = self.relation_embd(kg_input['relation_ids'])

        # Calculate GCN representations for concepts and relations, using 'num_hops' layers
        head_idx = kg_input['head_idx']
        tail_idx = kg_input['tail_idx']
        triple_labels = kg_input['triple_labels']
        node_repr, rel_repr = self._comp_gcn(
            concept_repr, 
            rel_repr,
            head_idx,
            tail_idx,
            triple_labels,
            layer_number=self.num_hops
        )

        # Construct triple representation
        head_repr = torch.gather(node_repr, 1, head_idx.unsqueeze(-1).expand(node_repr.size(0), head_idx.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1, tail_idx.unsqueeze(-1).expand(node_repr.size(0), tail_idx.size(1), node_repr.size(-1)))
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)

        logging.debug("\tShape of encoded triples: {}".format(triple_repr.shape))

        return triple_repr


    def _comp_gcn(self, concept_repr, rel_repr, head_idx, tail_idx, triple_labels, layer_number=2):
        '''
        concept_repr: B x Mc x E  (Mc=number of related concepts)
        rel_repr: B x Mt x E (Mt=number of triples)
        '''

        B = head_idx.size(0)
        Mt = head_idx.size(1)
        Mc = concept_repr.size(1)
        E = concept_repr.size(2)

        concept_hidden, relation_hidden = concept_repr, rel_repr
        for l in range(layer_number):

            # Initialise update_node for GCN calculation
            update_node = torch.zeros_like(concept_repr).to(concept_repr.device).float()
            count = torch.ones_like(head_idx).to(head_idx.device).masked_fill_(triple_labels == -1, 0).float()
            count_out = torch.zeros(B, Mc).to(head_idx.device).float()

            # Add the concept representations of the heads to node 'positions' of tails, subtract relation representation
            o = concept_hidden.gather(1, head_idx.unsqueeze(2).expand(B, Mt, E))
            o = o.masked_fill(triple_labels.unsqueeze(2) == -1, 0)
            scatter_add(o, tail_idx, dim=1, out=update_node)
            scatter_add(-relation_hidden.masked_fill(triple_labels.unsqueeze(2) == -1, 0), tail_idx, dim=1, out=update_node)
            scatter_add(count, tail_idx, dim=1, out=count_out)

            # Add the concept representations of the tails to node 'position' of heads, subtract relation representation
            o = concept_hidden.gather(1, tail_idx.unsqueeze(2).expand(B, Mt, E))
            o = o.masked_fill(triple_labels.unsqueeze(2) == -1, 0)
            scatter_add(o, head_idx, dim=1, out=update_node)
            scatter_add(-relation_hidden.masked_fill(triple_labels.unsqueeze(2) == -1, 0), head_idx, dim=1, out=update_node)
            scatter_add(count, head_idx, dim=1, out=count_out)

            # Combine calculated update to form new node and relation representations
            update_node = \
                self.W_s[l](concept_hidden) + \
                self.W_n[l](update_node) / count_out.clamp(min=1).unsqueeze(2)
            concept_hidden = self.act(update_node)
            relation_hidden = self.W_r[l](relation_hidden)

        return concept_hidden, relation_hidden


class KG_Probs_Model(nn.Module):

    def __init__(self, embedding_size, num_hops, gamma, aggregate_method, block_src, fixed_gate_value):

        self.embedding_size = embedding_size
        self.num_hops = num_hops
        self.gamma = gamma
        self.aggregate_method = aggregate_method
        self.block_src = block_src
        self.fixed_gate_value = fixed_gate_value

        super().__init__()
        self.triple_linear = nn.Linear(embedding_size * 3, embedding_size, bias=False)
        self.gate_linear = nn.Linear(embedding_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lm_hidden_states, lm_probs, triple_repr, kg_mem):
        '''
        return: 
            - probs: B x L x V
            - gate: B x L x 1
        '''

        # Combine hidden states with knowledge triples to calculate triple score
        triple_logits = torch.matmul(
            lm_hidden_states, 
            self.triple_linear(triple_repr).transpose(1, 2)
        )
        triple_prob = self.sigmoid(triple_logits)
        invalid_mask = (kg_mem["triple_labels"] == -1).unsqueeze(1)
        triple_prob = triple_prob.masked_fill(invalid_mask, 0)
        # B x L x Mt

        # Aggregate probability to nodes
        concept_scores = self._multi_hop(triple_prob, kg_mem)
        concept_probs = self.softmax(concept_scores)

        # Calculate probability for concepts
        index = kg_mem["vocab_map"].unsqueeze(1).expand(concept_probs.size(0), concept_probs.size(1), -1)
        concept_probs_vocab = concept_probs.gather(2, index)
        invalid_mask = (kg_mem["map_mask"] == 0).unsqueeze(1)
        concept_probs_vocab.masked_fill_(invalid_mask, 0)

        # Determine gate value (which determines whether to take token from language model or select a concept)
        if self.fixed_gate_value != None:
            gate = torch.ones((lm_probs.size(0), lm_probs.size(1) ,1), device=lm_probs.device) * self.fixed_gate_value
        else:
            gate = self.sigmoid(self.gate_linear(lm_hidden_states))

        # Determine combined token probabilities
        probs = gate * concept_probs_vocab + (1 - gate) * lm_probs
        is_concept = (torch.argmax(probs, dim=-1) != torch.argmax(lm_probs, dim=-1)).long()

        return probs, gate, concept_probs_vocab, triple_prob, is_concept

    def _multi_hop(self, triple_prob, kg_mem):
        '''
        triple_prob: B x L x Mt
        distance: B x Mc
        head, tail: B x Mt
        concept_label: B x Mc
        triple_label: B x Mt
        '''
        distance = kg_mem["distances"]

        # Init binary vector with source concept == 1 and others 0, and expand to size B, L, M
        concept_scores = []
        concept_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
        init_mask = torch.zeros_like(distance).unsqueeze(1).expand(*concept_size).to(distance.device).float()
        init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
        final_mask = init_mask.clone()
        init_mask.masked_fill_((kg_mem["concept_labels"] == -1).unsqueeze(1), 0)
        concept_scores.append(init_mask)

        head = kg_mem["head_idx"].unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = kg_mem["tail_idx"].unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)

        for _ in range(self.num_hops):

            # Calculate triple head score
            node_score = concept_scores[-1]
            triple_head_score = node_score.gather(2, head)
            triple_head_score.masked_fill_((kg_mem["triple_labels"] == -1).unsqueeze(1), 0)
            
            # Aggregate scores to tail nodes
            update_value = triple_head_score * self.gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if self.aggregate_method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif self.aggregate_method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)
            out.masked_fill_((kg_mem["concept_labels"] == -1).unsqueeze(1), 0)           
            concept_scores.append(out)
             
        total_concept_score = final_mask
        if self.block_src:
            total_concept_score *= -1e5     # Punish start-nodes by assigning large negative value
        for score in concept_scores[1:]:
            total_concept_score += score
        # B x L x Mc

        return total_concept_score


class KG_LogitsProcessor(LogitsProcessor):

    def __init__(self, gpt2model, kg_probs, inputs, triple_repr, kg_mem):

        self.gpt2model = gpt2model
        self.kg_probs = kg_probs
        self.triple_repr = triple_repr
        self.kg_mem = kg_mem
        self.call_count = 0
        self.input_repr = self.gpt2model(
            input_ids = inputs.input_ids[:, :-1],
            attention_mask = inputs.attention_mask[:, :-1],
            output_hidden_states=True
        )
        self.attention_mask = inputs.attention_mask
        self.call_count += 1

    def __call__(self, input_ids, scores):

        self.call_count += 1
        logging.spam("Call: {}\n{}".format(self.call_count, input_ids))
        B = input_ids.shape[0]
        dev = input_ids.device

       # Calculate probabilities according to language model
        lm_output = self.gpt2model(
            input_ids=input_ids[:, -1].view(-1, 1), 
            past_key_values=self.input_repr.past_key_values,
            attention_mask=self.attention_mask,
            output_hidden_states=True, 
            output_attentions=True
        )
        lm_hidden_states = lm_output.hidden_states[-1]
        lm_probs = nn.functional.softmax(scores, dim=-1).unsqueeze(dim=1)

        # Combine hidden states with knowledge triples to calculate adjusted probabilities
        probs, gate, concept_probs_vocab, triple_prob, is_concept = self.kg_probs(lm_hidden_states, lm_probs, self.triple_repr, self.kg_mem)

        self.input_repr = lm_output
        self.attention_mask = torch.cat([self.attention_mask, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=1)
        kg_scores = probs.exp().squeeze()

        return kg_scores    # Note: these are not typical logits, but combined and exponentiated probabilities
    

class KnowledgeGroundedDecoder(nn.Module):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('KnowledgeGroundedDecoder')
        group.add_argument(
            '--embedding_size', type=int, default=768, help='Hidden size.'
        )
        group.add_argument(
            "--num_hops",
            type=int,
            default=2,
            help="Number of hops in the graph to look for related concepts."
        )
        group.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="Parameter for impact of gate loss in loss calculation."
        )
        group.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="Parameter for impact of triple loss in loss calculation."
        )
        group.add_argument(
            "--gamma",
            type=float,
            default=0.8,
            help="Parameter for calculation of probabilities of triple heads"
        )
        group.add_argument(
            "--gate",
            type=float,
            default=None,
            help="If set, uses a fixed gate probability [0.0 - 1.0]"
        )
        group.add_argument(
            "--aggregate_method",
            type=str,
            default="max",
            choices=["avg", "max"],
            help="How to aggregate probabilities on graph nodes."
        )
        group.add_argument(
            "--fixed_lm",
            type=bool,
            default=False,
            help="Freeze the weights of the GPT2 language model during training."
        )
        group.add_argument(
            "--block-src",
            type=bool,
            default=True,
            help="Blocking source concepts in reasoning stimulates generation of new related concepts."
        )
        return parser
    
    def __init__(self, opt, tokenizer):
        super().__init__()

        # Model and parameters for language model
        self.gpt2model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2model.resize_token_embeddings(len(tokenizer))
        if opt['fixed_lm'] == True:
            self.fix_lm_weights()

        # Model and parameters to encode knowledge triples
        self.triple_encoder = TripleEncoder(self.gpt2model.transformer.wte, opt['num_hops'])

        # Graph convolutionel network to calculate concept probabilities
        self.kg_probs = KG_Probs_Model(
            opt['embedding_size'],
            opt['num_hops'],
            opt['gamma'],
            opt['aggregate_method'],
            opt['block_src'],
            opt['gate']
        )

        # Gate to control generation via language model or knowledge model
        self.fixed_gate_value = opt['gate']
        self.gate_linear = nn.Linear(opt['embedding_size'], 1)
        self.sigmoid = nn.Sigmoid()

        logging.info("Initialized KnowledgeGroundedDecoder")

    def fix_lm_weights(self):
        for param in self.gpt2model.parameters():
            param.requires_grad = False
    
    def forward(self, inputs, decoder_input, kg_input):

        logging.debug("Forward KnowledgeGroundedDecoder")

        # Calculate probabilities according to language model
        lm_output = self.gpt2model(
            input_ids=torch.cat([inputs.input_ids, decoder_input.input_ids], dim=1),
            attention_mask=torch.cat([inputs.attention_mask, decoder_input.attention_mask], dim=1),
            output_hidden_states=True, 
            output_attentions=True
        )
        lm_hidden_states = lm_output.hidden_states[-1]
        lm_probs = self.softmax(lm_output.logits)

        # Calculate triple representations
        triple_repr = self.triple_encoder(kg_input)
        kg_input["triple_repr"] = triple_repr

        # Combine hidden states with knowledge triples to calculate adjusted probabilities
        probs, gate, concept_probs_vocab, triple_prob, is_concept = self.kg_probs(lm_hidden_states, lm_probs, kg_input)

        return probs, gate, lm_probs, concept_probs_vocab, triple_prob, is_concept
    

    def generate(self, batch, device):
    
        logging.debug("NEW Generate KnowledgeGroundedDecoder")
     
        inputs = batch['inputs'].to(device)
        kg_mem = self._kg_input(batch, device)
        triple_repr = self.triple_encoder(kg_mem)
        kg_logits_processor = KG_LogitsProcessor(self.gpt2model.transformer, self.kg_probs, inputs, triple_repr, kg_mem)

        preds = self.gpt2model.generate(
            **inputs,
            num_beams=1, do_sample=False,
            return_dict_in_generate=False,
            output_scores=False,
            max_new_tokens=20,
            logits_processor=LogitsProcessorList([kg_logits_processor]), 
        )
        return preds

    def _kg_input(self, batch, device):
        return {
            'concept_ids': batch['concept_ids'].to(device),
            'relation_ids': batch['relation_ids'].to(device),
            'distances': batch['distances'].to(device),
            'head_idx': batch['head_idx'].to(device),
            'tail_idx': batch['tail_idx'].to(device),
            'concept_labels': batch['concept_labels'].to(device),
            'triple_labels': batch['triple_labels'].to(device),
            'vocab_map': batch['vocab_map'].to(device),
            'map_mask': batch['map_mask'].to(device)
        }

    def train_step(self, batch, optimizer, criterion, device):

        inputs = batch['inputs'].to(device)
        labels = batch['labels'].to(device) 
        kg_input = self._kg_input(batch, device)
    
        optimizer.zero_grad()
        probs, gate, lm_probs, concept_probs, triple_prob, is_concept = self.forward(inputs, labels, kg_input)
        loss, gen_loss, triple_loss, gate_loss = criterion(
            probs, labels.input_ids, 
            triple_prob, batch['triple_labels'], 
            gate, batch['gate_labels']
        )
        logging.debug("Train: loss {:.4f}, gen_loss {:.4f}, triple_loss {:.4f} gate_loss {:.4f}".format(
            loss.mean().item(), 
            gen_loss.mean().item(), 
            triple_loss.mean().item(),
            gate_loss.mean().item()
        ))
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    

    def valid_step(self, batch, criterion, device):

        inputs = batch['inputs'].to(device)
        labels = batch['labels'].to(device) 
        kg_input = self._kg_input(batch, device)
    
        with torch.no_grad():
            probs, gate, lm_probs, concept_probs, triple_prob, is_concept = self.forward(inputs, labels, kg_input)
            loss, gen_loss, triple_loss, gate_loss = criterion(
                probs, labels.input_ids, 
                triple_prob, batch['triple_labels'], 
                gate, batch['gate_labels']
            )

        pred = probs.cpu().argmax(dim=-1)
        ys = ys.cpu()
        ignore_mask = ys.ne(criterion.ignore_index)

        # LM accuracy
        token_correct = ys.eq(pred) * ignore_mask
        token_acc = (token_correct.sum() / ignore_mask.sum()).item() 

        stats = {
            "loss": loss.mean().item(),
            "token_prediction_acc": token_acc
        }

        logging.debug("Valid: loss {:.4f}, gen_loss {:.4f}, triple_loss {:.4f} gate_loss {:.4f}".format(
            loss.mean().item(), 
            gen_loss.mean().item(), 
            triple_loss.mean().item(),
            gate_loss.mean().item()
        ))
        return stats


if __name__ == "__main__":

    ###
    ### Test triple encoder
    ###
    
    e = torch.tensor(range(10)).unsqueeze(dim=1).expand(10,8).float()
    embedding = nn.Embedding.from_pretrained(e)
    num_hops=2

    triple_encoder = TripleEncoder(embedding=embedding, num_hops=num_hops)

    # Change inner parameters
    r = torch.tensor(range(40)).unsqueeze(dim=1).expand(40,8).float()
    triple_encoder.relation_embd = nn.Embedding.from_pretrained(r)
    triple_encoder.W_s = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
    triple_encoder.W_r = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
    triple_encoder.W_n = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 


    concept_ids = torch.tensor([[0, 1, 4, 5, 7, 2, 8, 6]])
    relation_ids = torch.tensor([[1, 1, 2, 3, 1, 1, 2, 3, 3]])
    head_idx = torch.tensor([[0, 0, 0, 0, 1, 4, 4, 3, 2]])
    tail_idx = torch.tensor([[1, 4, 3, 2, 5, 6, 3, 7, 7]])
    triple_labels = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 0, 1]])
    kg_input = {
        'concept_ids': concept_ids,
        'relation_ids': relation_ids,
        'head_idx': head_idx,
        'tail_idx': tail_idx,
        'triple_labels': triple_labels
    }
    encoding = triple_encoder(kg_input)
    print(encoding)

    ###
    ### Test KnowledgeGroundedDecoder
    ###

    e = torch.tensor(range(10)).unsqueeze(dim=1).expand(10,768).float()
    embedding = nn.Embedding.from_pretrained(e)
    num_hops=2

    triple_encoder = TripleEncoder(embedding=embedding, num_hops=num_hops)

    # Change inner parameters
    r = torch.tensor(range(40)).unsqueeze(dim=1).expand(40,768).float()
    triple_encoder.relation_embd = nn.Embedding.from_pretrained(r)
    triple_encoder.W_s = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
    triple_encoder.W_r = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
    triple_encoder.W_n = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 

    c2i = {"A":0, "F":1, "G": 2, "N": 3, "B": 4, "H":5, "C":6, "E":7, "D":8, "K":9}
    i2c = ["A", "F", "G", "N", "B", "H", "C", "E", "D", "K"]

    concept_ids = torch.tensor([[0, 1, 4, 5, 7, 2, 8, 6]])
    relation_ids = torch.tensor([[1, 1, 2, 3, 1, 1, 2, 3, 3]])
    head_ids = torch.tensor([[0, 0, 0, 0, 1, 4, 4, 3, 2]])
    tail_ids = torch.tensor([[1, 4, 3, 2, 5, 6, 3, 7, 7]])
    triple_labels = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 0, 1]])
    kg_input = {
        'concept_ids': concept_ids,
        'relation_ids': relation_ids,
        'head_idx': head_ids,
        'tail_idx': tail_ids,
        'triple_labels': triple_labels
    }

    encoding = triple_encoder(kg_input)
    print(encoding)

    opt = {
        "num_hops": 2,
        "aggregate_method": "max",
        "embedding_size": 768,
        "max_concepts": 400,
        "max_triples": 800,
        "alpha": 0.7,
        "beta": 0.2,
        "gamma": 0.33,
        'hidden_size': 256,
    }

    # model = KnowledgeGroundedDecoder(opt)
