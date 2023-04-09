import torch
import torch.nn as nn

import utils.logging as logging
from utils.general import padded_tensor

from transformers import BartForConditionalGeneration, BartModel, BartConfig, GenerationConfig
from torchmetrics.functional.text.perplexity import perplexity


BART_BASE = 'facebook/bart-large-cnn'

class ConditionalFactLoss(nn.Module):

    def __init__(self, nofact_token_id, ignore_index=-100, lm_weight=0.5):
        super().__init__()
        self.nofact_token_id = nofact_token_id
        self.nllloss = nn.NLLLoss(ignore_index=ignore_index, reduction='none')
        self.lm_weight = lm_weight

    def forward(self, fact_logprobs, lm_logprobs, target):

        assert (lm_logprobs.shape[2] > 0) and (target.shape[1] > 0), "Invalid shape for lm_logprobs {} or target {}".format(input.shape, target.shape)

        # Classification loss: whether facts are recognized correctly
        target_fact = (target[:, 1] != self.nofact_token_id).int()
        classification_loss = -fact_logprobs[:, 0] * (1 - target_fact) - fact_logprobs[:, 1]  * target_fact

        # LM loss: whether the tokens of the facts are predicted correctly
        lm_loss = self.nllloss(lm_logprobs, target).mean(dim=1)

        # Weighted combination of classification loss and LM loss
        combined_loss = (1 - target_fact) * classification_loss + target_fact * (self.lm_weight * lm_loss + (1 - self.lm_weight) * classification_loss)
        
        return combined_loss.mean(), classification_loss.mean(), lm_loss.mean()

class BartExtractor(nn.Module):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('BartExtractor')
        group.add_argument("--lm_loss_factor", type=float, default=0.5, help="Relative weight of lm_loss in combined loss")
        group.add_argument("--teacher_forcing", default=False, action='store_true', help="Use teacher forcing")
        group.add_argument("--decoder_max", type=int, default=50, help="Max number of tokens to generate")
        return parser

    def __init__(self, bart_base=None, nofact_token_id=None):
        super().__init__()
        assert nofact_token_id is not None, "Must provide nofact_token_id"
        self.nofact_token_id = nofact_token_id
        if bart_base is None:
            self.bart = BartForConditionalGeneration(config=BartConfig())
        else:
            self.bart = BartForConditionalGeneration.from_pretrained(bart_base)
        self.nofact_sequence = torch.tensor([self.bart.config.decoder_start_token_id, self.bart.config.bos_token_id, self.nofact_token_id, self.bart.config.eos_token_id])
        if self.nofact_token_id == self.bart.config.eos_token_id:
            self.nofact_sequence = self.nofact_sequence[:-1]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def select_fact_logprobs(self, lm_logprobs):
        logprob_nofact = lm_logprobs[:, self.nofact_token_id]
        all_tokens_ids = torch.arange(lm_logprobs.shape[1])
        all_tokens_ids_except_nofact = all_tokens_ids[all_tokens_ids != self.nofact_token_id]
        logprob_fact = lm_logprobs[:, all_tokens_ids_except_nofact].max(dim=1)[0]
        fact_logprobs = torch.stack([logprob_nofact, logprob_fact], dim=1)
        return fact_logprobs
    
    def forward(self, input_ids, attention_mask, labels):
        lm_logits = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        lm_logprobs = self.logsoftmax(lm_logits)
        fact_logprobs = self.select_fact_logprobs(lm_logprobs[:, 1, :])  # Token directly after <bos> token signals whether sequence is a fact
        return fact_logprobs, lm_logprobs
    
    def generate(self, input_ids, **kwargs):

        # # determine relevant length of each sequence
        # relevant = kwargs.get("attention_mask")
        # if relevant is None:
        #     relevant = input_ids != self.bart.config.pad_token_id
        # else:
        #     del kwargs['attention_mask']
        # input_lengths = relevant.sum(dim=1)

        # # generate output tokens for earch input sequence individually
        # gen_out = padded_tensor(
        #     [
        #         self.bart.generate(input_sequence[:l].unsqueeze(dim=0), **kwargs).squeeze(dim=0)
        #         for input_sequence, l in zip(input_ids, input_lengths)
        #     ],
        #     pad_value=self.bart.config.pad_token_id
        # )
        gen_out = self.bart.generate(input_ids, **kwargs)
        pred_fact = gen_out[:, 2] != self.nofact_token_id

        logging.spam("Generate: pred_fact={}".format(pred_fact))
        logging.spam("Generate: gen_out={}".format(gen_out))

        gen_out_cleaned = padded_tensor(
            [
                gen if is_fact else self.nofact_sequence
                for is_fact, gen in zip(pred_fact, gen_out)
            ],
            pad_value=self.bart.config.pad_token_id
        )
        return gen_out_cleaned

    def train_step(self, batch, optimizer, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)

        optimizer.zero_grad()
        fact_logprobs, lm_logprobs = self.forward(input_ids, attention_mask, y)
        loss, classification_loss, lm_loss = criterion(fact_logprobs, lm_logprobs.transpose(1,2), y)
        logging.debug("Train: loss {:.4f}, cls_loss {:.4f}, lm_loss {:.4f}".format(loss, classification_loss, lm_loss))
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def valid_step(self, batch, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)

        with torch.no_grad():
            fact_logprobs, lm_logprobs = self.forward(input_ids, attention_mask, y)
            loss, classification_loss, lm_loss = criterion(fact_logprobs, lm_logprobs.transpose(1,2), y)

        pred = lm_logprobs.cpu().argmax(dim=-1)
        ignore_mask = batch['labels'].ne(self.bart.config.pad_token_id)
        
        # Classification accuracy
        pred_fact = pred[:, 1] != self.nofact_token_id  # Check for nofact-token, directly after the start-of-sentence
        label_fact = batch['labels'][:, 1] != self.nofact_token_id
        fact_correct = label_fact.eq(pred_fact)
        fact_acc = fact_correct.sum().item() / batch['labels'].shape[0]

        # LM accuracy
        token_correct = batch['labels'].eq(pred) * ignore_mask
        token_acc = (token_correct.sum() / ignore_mask.sum()).item() 

        # LM perplexity
        ppl = perplexity(preds=lm_logprobs, target=y, ignore_index=self.bart.config.pad_token_id)

        stats = {
            "loss": loss.item(),
            "classification_loss": classification_loss.item(),
            "lm_loss": lm_loss.item(),
            "acc": fact_acc,
            "perplexity": ppl, 
            "token_prediction_acc": token_acc
        }
        logging.debug("Valid: loss {:.4f}, cls_loss {:.4f}, lm_loss {:.4f}, cls_acc {:.4f}, lm_acc {:.4f}, ppl {:.4f}".format(loss, classification_loss, lm_loss, fact_acc, token_acc, ppl))

        return stats


class PrefixBart(BartExtractor):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('PrefixBart')
        group.add_argument("--freeze", type=int, default=0, help="Layers to freeze for finetuning; None=none, 0=only embeddings, 12=all")
        group.add_argument("--enc_prefix_size", type=int, default=0, help="Insert prefix in BART encoder")
        group.add_argument("--dec_prefix_size", type=int, default=0, help="Insert prefix in BART decoder")
        group.add_argument("--lm_loss_factor", type=float, default=0.5, help="Relative weight of lm_loss in combined loss")
        group.add_argument("--prefix_aggr", type=str, default="concat", choices=["concat", "max", "avg"], help="How to aggregate prefix hidden states")
        group.add_argument("--teacher_forcing", default=False, action='store_true', help="Use teacher forcing")
        group.add_argument("--decoder_max", type=int, default=50, help="Max number of tokens to generate")
        return parser

    def __init__(self, bart_base=None, nofact_token_id=None, freeze=None, enc_prefix_size=1, dec_prefix_size=1, lm_loss_factor=0.5, prefix_aggr="concat"):
        super().__init__(bart_base=bart_base, nofact_token_id=nofact_token_id)
        self.enc_prefix_size = enc_prefix_size
        self.dec_prefix_size = dec_prefix_size
        self.lm_loss_factor = lm_loss_factor
        self.prefix_aggr = prefix_aggr
        if self.enc_prefix_size > 0:
            self.enc_prefix_ids = torch.arange(self.enc_prefix_size)
            self.enc_prefix = nn.Embedding(self.enc_prefix_size, self.bart.config.d_model)
        if self.dec_prefix_size > 0:
            self.dec_prefix_ids = torch.arange(self.dec_prefix_size)
            self.dec_prefix = nn.Embedding(self.dec_prefix_size, self.bart.config.d_model)
        classifier_input_size = self.bart.config.d_model * (enc_prefix_size + 1 if prefix_aggr == "concat" else 1)
        self.classifier = nn.Linear(in_features=classifier_input_size, out_features=2)

        if freeze is None:
            modules = []
        else:
            modules = [
                self.bart.model.shared,
                self.bart.model.encoder.embed_tokens,
                self.bart.model.encoder.embed_positions,
                self.bart.model.encoder.layernorm_embedding,
                self.bart.model.decoder.embed_tokens,
                self.bart.model.decoder.embed_positions,
                self.bart.model.decoder.layernorm_embedding,
                *(self.bart.model.encoder.layers[:freeze]), 
                *(self.bart.model.decoder.layers[:freeze])
            ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def encode(self, input_ids, attention_mask):

        B, L = input_ids.shape
        trunc_L = self.bart.config.max_position_embeddings - self.enc_prefix_size
        dev = input_ids.device

        # prepend prefix to input embeddings
        if self.enc_prefix_size > 0:
            enc_prefix_embeddings = self.enc_prefix(self.enc_prefix_ids.to(dev)).unsqueeze(dim=0).expand(B, -1, -1)
        else:
            enc_prefix_embeddings = torch.tensor([], device=dev)
        input_embeddings = self.bart.model.shared(input_ids)[:, :trunc_L, :]
        input_embeddings = torch.cat([enc_prefix_embeddings, input_embeddings], dim=1)

        # adjust attention mask
        if attention_mask is not None:
            enc_prefix_attention_mask = torch.ones((B, self.enc_prefix_size), dtype=torch.long, device=dev)
            attention_mask = torch.cat([enc_prefix_attention_mask, attention_mask[:, :trunc_L]], dim=1)

        # pool last hidden states of prefix with cls
        encoded = self.bart.model.encoder(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        cls = encoded.last_hidden_state[:, :self.enc_prefix_size + 1, :]
        if self.prefix_aggr == "concat":
            cls = cls.view(B, -1)
        elif self.prefix_aggr == "max":
            cls = cls.max(dim=1)[0]
        elif self.prefix_aggr == "avg":
            cls = cls.mean(dim=1)

        # use pooled cls for classification of fact
        fact_logits = self.classifier(cls)
        fact_logprobs = self.logsoftmax(fact_logits)

        return encoded, fact_logprobs

    def forward(self, input_ids, attention_mask, labels):

        # encode input and calculate probability it contains a fact
        encoded, fact_logprobs = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        # calculate decoder output based on encoded input_ids
        lm_logits = self.bart(encoder_outputs=encoded, labels=labels).logits
        lm_logprobs = self.logsoftmax(lm_logits)

        return fact_logprobs, lm_logprobs


    def generate(self, input_ids, **kwargs):

        # Encode input and calculate probability it contains a fact
        encoded, fact_logprobs = self.encode(input_ids=input_ids, attention_mask=None)
        pred_fact = fact_logprobs.argmax(dim=-1)
        logging.spam("Generate: pred_fact={}".format(pred_fact))

        # Generate fact if model predicts input contains facts
        nofact_sequence = torch.tensor([self.bart.config.decoder_start_token_id, self.bart.config.bos_token_id, self.nofact_token_id])
        gen_out = torch.stack([
            self.bart.generate(inputs_embeds=e.unsqueeze(dim=0), **kwargs)[0] if is_fact else nofact_sequence
            for is_fact, e in zip(pred_fact, encoded.last_hidden_state)
        ])
        logging.spam("Generate: gen_out={}".format(gen_out))

        return gen_out

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from dataset.msc_summary_turns import MSC_Turns

    logging.set_log_level(logging.SPAM)
    logging.set_only_message(True)

    # Settings for test
    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/'
    sessions = [1]
    subset = 'train'
    len_context = 2
    speaker_prefixes = None #["<me>", "<you>"]
    nofact_token = '' #'<nofact>'
    add_tokens = None #speaker_prefixes + [nofact_token]
    model_class = "bart"
    lm_loss_factor = 0.5
    freeze=None
    enc_prefix_size=0
    dec_prefix_size=0
    prefix_aggr="concat"

    # Setup
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    if add_tokens is not None:
        num_added_toks = tokenizer.add_tokens(add_tokens)
    nofact_token_id = tokenizer.convert_tokens_to_ids(nofact_token) if nofact_token != '' else tokenizer.eos_token_id
    assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(nofact_token)

    if model_class == "bart":
        model = BartExtractor(bart_base="facebook/bart-base", nofact_token_id=nofact_token_id)
    else:
        model = PrefixBart(
            bart_base="facebook/bart-base", 
            nofact_token_id=nofact_token_id, 
            freeze=freeze, 
            enc_prefix_size=enc_prefix_size,
            dec_prefix_size=dec_prefix_size,
            prefix_aggr=prefix_aggr
        )
    model.bart.resize_token_embeddings(len(tokenizer))
    criterion = ConditionalFactLoss(nofact_token_id=nofact_token_id, ignore_index=tokenizer.pad_token_id, lm_weight=lm_loss_factor)

    msc_turns = MSC_Turns(basedir=basedir, sessions=sessions, subset=subset, tokenizer=tokenizer, len_context=len_context, speaker_prefixes=speaker_prefixes, nofact_token=nofact_token)
    data = [msc_turns[i] for i in range(10)]
    batch = msc_turns.batchify(data)

    # Test forward
    fact_logprobs, lm_logprobs = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
    pred_fact = fact_logprobs.argmax(dim=-1) 
    preds = lm_logprobs.argmax(dim=-1)
    response = tokenizer.batch_decode(preds)

    for i in range(3):
        print('-' * 40)
        print(data[i][0])
        print(data[i][1])
        print(response[i] if pred_fact[i] else nofact_token)

    # Test loss function and valid_step
    valid_stats = model.valid_step(batch, criterion, device="cpu")
    logging.report("valid_stats = {}".format(valid_stats))

    # Test generate
    with torch.no_grad():
        pred_tokens = model.generate(batch['input_ids'], max_new_tokens=20)
    pred_persona = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("GENERATE")
    for i, (t, p) in enumerate(zip(pred_tokens, pred_persona)):
        print('-' * 40)
        print('context:    ', data[i][0])
        print('target:     ', data[i][1])
        print('tokens:     ', t)
        print('prediction: ', p)




