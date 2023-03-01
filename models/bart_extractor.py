import torch
import torch.nn as nn

import utils.logging as logging

from transformers import BartForConditionalGeneration, BartModel, BartConfig, GenerationConfig
from transformers.models.bart.modeling_bart import shift_tokens_right


class ConditionalFactLoss(nn.Module):

    def __init__(self, nofact_token_id, ignore_index=-100, lm_weight=0.5):
        super().__init__()
        self.nofact_token_id = nofact_token_id
        self.nllloss = nn.NLLLoss(ignore_index=ignore_index, reduction='none')
        self.lm_weight = lm_weight

    def forward(self, input, target):

        assert (input.shape[2] > 0) and (target.shape[1] > 0), "Invalid shape for input {} or target {}".format(input.shape, target.shape)

        # Classification loss: whether facts are recognized correctly
        logprob_nofact = input[:, self.nofact_token_id, 1]
        predict_nofact = ((input[:, :, 1] > logprob_nofact.unsqueeze(1)).sum(dim=1) == 0).int()  # No token has higher prob than 'no_fact_token'
        target_nofact = (target[:, 1] == self.nofact_token_id).int()

        all_tokens_ids = torch.arange(input.shape[1])
        all_tokens_ids_except_nofact = all_tokens_ids[all_tokens_ids != self.nofact_token_id]
        logprob_fact = input[:, all_tokens_ids_except_nofact, 1].max(dim=1)[0]

        classification_loss = -logprob_nofact * target_nofact - logprob_fact  * (1 - target_nofact)
        correct_fact = predict_nofact.eq(target_nofact)

        # LM loss: whether the tokens of the facts are predicted correctly
        lm_loss = self.nllloss(input, target).mean(dim=1)

        combined_loss = target_nofact * classification_loss + (1 - target_nofact) * (self.lm_weight * lm_loss + (1 - self.lm_weight) * classification_loss)
        
        return combined_loss.mean(), classification_loss.mean(), lm_loss.mean()

class BartExtractor(nn.Module):

    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument("--teacher_forcing", default=False, action='store_true', help="Use teacher forcing")
        parser.add_argument("--decoder_max", type=int, default=50, help="Max number of tokens to generate")
        return parser

    def __init__(self, bart_base=None, nofact_token_id=None):
        super().__init__()
        assert nofact_token_id is not None, "Must provide nofact_token_id"
        self.nofact_token_id = nofact_token_id
        if bart_base is None:
            self.bart = BartForConditionalGeneration(config=BartConfig())
        else:
            self.bart = BartForConditionalGeneration.from_pretrained(bart_base)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        # self.gen_config = GenerationConfig(
        #     min_length=3,
        #     min_new_tokens=3,
        #     early_stopping=True,
        #     no_repeat_ngram_size=3,
        #     num_beams=1,
        #     do_sample=False
        # )

    def forward(self, input_ids, attention_mask, labels):
        logits = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        logprobs = self.logsoftmax(logits)
        return logprobs

    def train_step(self, batch, optimizer, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        optimizer.zero_grad()
        
        logits = self.forward(input_ids, attention_mask, y)
        loss, classification_loss, lm_loss = criterion(logits.transpose(1,2), y)
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def valid_step(self, batch, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, y)
            loss, classification_loss, lm_loss = criterion(logits.transpose(1,2), y)

        pred = logits.cpu().argmax(dim=-1)
        ignore_mask = batch['labels'].ne(self.bart.config.pad_token_id)
        pred_fact = pred[:, 1] != self.nofact_token_id
        label_fact = batch['labels'][:, 1] != self.nofact_token_id
        fact_correct = pred_fact == label_fact
        fact_acc = fact_correct.sum().item() / batch['labels'].shape[0]
        token_correct = batch['labels'].eq(pred) * ignore_mask
        token_acc = (token_correct.sum() / ignore_mask.sum()).item() 

        stats = {
            "loss": loss.item(),
            "classification_loss": classification_loss,
            "lm_loss": lm_loss,
            "acc": fact_acc,
            "token_prediction_acc": token_acc
        }

        return stats


class PrefixBart(BartForConditionalGeneration):


    def __init__(self, bert_base=None, freeze=None, prefix_size=1):
        super().__init__(bert_base, freeze)
        self.prefix_size = prefix_size
        if self.prefix_size > 0:
            self.prefix_ids = torch.arange(self.prefix_size)
            self.prefix = nn.Embedding(self.prefix_size, self.bert.config.hidden_size)

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
        cls = cls.max(dim=1)[0]

        # use pooled cls for classification
        logits = self.classifier(cls)
        logprobs = self.softmax(logits)

        return logprobs


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from eval import print_bart_predictions

    logging.set_log_level(logging.DEBUG)
    logging.set_only_message(True)

    data = {
        'history': [
            "What kind of car do you own? I have a jeep.", 
            "What kind of car do you own? I have a jeep.", 
            "I'm a computer programmer. What do you do for work."
        ],
        'last_utterance': [
            "I don't own my own car! I actually really enjoying walking and running, but then again, I live in a small town and semi-close to work.",
            "What kind of jeep?", 
            "I work in marketing. Do you have any hobbies?"
        ],
        'personas': [
            "I live semi-close to work. I don't own a car. I enjoy running and walking. I live in a small town.", 
            "",
            "I have a marketing job."
        ]
    }

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = BartExtractor()
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    loss_fn = ConditionalFactLoss(nofact_token_id=tokenizer.eos_token_id, ignore_index=tokenizer.pad_token_id)

    batch = tokenizer(data['history'], data['last_utterance'], text_target=data['personas'], padding=True, return_tensors="pt")

    valid_stats = model.valid_step(batch, loss_fn, device="cpu")

    logits = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
    loss = loss_fn(logits.transpose(1,2), batch['labels'])
    preds = logits.argmax(dim=-1)
    response = tokenizer.batch_decode(preds)

    for i in range(3):
        print('-' * 40)
        print(data['history'][i] + ' / ' + data['last_utterance'][i])
        print(data['personas'][i])
        print(response[i])
    # logging.info(response)
    
    with torch.no_grad():
        pred_tokens = model.bart.generate(batch['input_ids'], max_new_tokens=20)
    pred_persona = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    for p in pred_persona:
        print('-' * 40)
        print(p)

    stats = model.valid_step(batch, criterion, "cpu")
    logging.report(stats)

