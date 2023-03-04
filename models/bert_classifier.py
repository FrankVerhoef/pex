import torch
import torch.nn as nn

import utils.logging as logging

from transformers import BertForSequenceClassification, BertModel, BertConfig


class BertClassifier(nn.Module):

    def __init__(self, classifier_size_factor=1, bert_base=None):
        super().__init__()
        if bert_base is None:
            self.bert = BertModel(config=BertConfig())
        else:
            self.bert = BertModel.from_pretrained(bert_base)
        self.bert.pooler = None

        in_features = classifier_size_factor * self.bert.config.hidden_size
        self.classifier = nn.Linear(in_features=in_features, out_features=2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        cls = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        probs = self.softmax(logits)
        return probs

    def train_step(self, batch, optimizer, criterion, device):

        (input_ids, attention_mask, token_type_ids), y = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def valid_step(self, batch, criterion, device):

        (input_ids, attention_mask, token_type_ids), y = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, y)

        pred = logits.cpu().argmax(dim=1)
        acc = pred.eq(y.cpu()).sum().item() / len(y)

        stats = {
            "loss": loss.item(),
            "acc": acc
        }

        return stats


class PrefixBert(BertClassifier):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('PrefixBert')
        group.add_argument("--freeze", type=int, default=0, help="Layers to freeze for finetuning; None=none, 0=only embeddings, 12=all")
        group.add_argument("--prefix_size", type=int, default=0, help="Insert prefix in BERT")
        group.add_argument("--prefix_aggr", type=str, default="concat", choices=["concat", "max", "avg"], help="How to aggregate prefix hidden states")
        return parser

    def __init__(self, bert_base=None, freeze=None, prefix_size=1, prefix_aggr="concat"):
        
        self.freeze = freeze
        self.prefix_size = prefix_size
        self.prefix_aggr = prefix_aggr
        classifier_size_factor = prefix_size + 1 if prefix_aggr == "concat" else 1
        super().__init__(classifier_size_factor, bert_base)
        if self.prefix_size > 0:
            self.prefix_ids = torch.arange(self.prefix_size)
            self.prefix = nn.Embedding(self.prefix_size, self.bert.config.hidden_size)
        if freeze is None:
            modules = []
        else:
            modules = [self.bert.embeddings, *(self.bert.encoder.layer[:freeze])]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):

        B, L = input_ids.shape
        trunc_L = self.bert.config.max_position_embeddings - self.prefix_size
        dev = input_ids.device

        # prepend prefix to input embeddings
        if self.prefix_size > 0:
            prefix_embeddings = self.prefix(self.prefix_ids.to(dev)).unsqueeze(dim=0).expand(B, -1, -1)
        else:
            prefix_embeddings = torch.tensor([], device=dev)
        input_embeddings = self.bert.embeddings.word_embeddings(input_ids)[:, :trunc_L, :]
        input_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)

        # adjust attention mask and token_type_ids
        prefix_attention_mask = torch.ones((B, self.prefix_size), dtype=torch.long, device=dev)
        prefix_token_type_ids = torch.zeros((B, self.prefix_size), dtype=torch.long, device=dev)
        attention_mask = torch.cat([prefix_attention_mask, attention_mask[:, :trunc_L]], dim=1)
        token_type_ids = torch.cat([prefix_token_type_ids, token_type_ids[:, :trunc_L]], dim=1)

        # pool last hidden states of prefix with cls
        out = self.bert(inputs_embeds=input_embeddings, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, :self.prefix_size + 1, :]
        if self.prefix_aggr == "concat":
            cls = cls.view(B, -1)
        elif self.prefix_aggr == "max":
            cls = cls.max(dim=1)[0]
        elif self.prefix_aggr == "avg":
            cls = cls.mean(dim=1)

        # use pooled cls for classification
        logits = self.classifier(cls)
        logprobs = self.softmax(logits)

        return logprobs


if __name__ == "__main__":

    logging.set_log_level(logging.DEBUG)
    logging.set_only_message(True)

    model = PrefixBert(prefix_size=3)
    criterion = nn.NLLLoss()
    input_ids = torch.randint(high=100, size=(3,8))
    attention_mask = torch.randint(high=2, size=(3,8))
    tokentype_ids = torch.randint(high=2, size=(3,8))
    X = (input_ids, attention_mask, tokentype_ids)
    y = torch.randint(high=2, size=(3,))

    logging.info(X)
    logging.info(y)

    outputs = model(*X)
    logging.report(outputs)

    stats = model.valid_step((X,y), criterion, "cpu")
    logging.report(stats)