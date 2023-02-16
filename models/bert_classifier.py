import torch
import torch.nn as nn

from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids).logits
        logits = self.softmax(out)
        return logits

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

if __name__ == "__main__":

    model = BertClassifier()
    criterion = nn.NLLLoss()
    input_ids = torch.randint(high=100, size=(3,8))
    attention_mask = torch.randint(high=2, size=(3,8))
    tokentype_ids = torch.randint(high=2, size=(3,8))
    X = (input_ids, attention_mask, tokentype_ids)
    y = torch.randint(high=2, size=(3,))

    print(X)
    print(y)

    outputs = model(*X)

    print(outputs)

    stats = model.valid_step((X,y), criterion, "cpu")
    print(stats)