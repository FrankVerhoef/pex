import torch
import torch.nn as nn

import utils.logging as logging

from transformers import BartForConditionalGeneration, BartConfig
from torchmetrics.functional.text.perplexity import perplexity


class ExtractedFactLoss(nn.Module):

    def __init__(self, nofact_token_id, ignore_index=-100, lm_weight=0.5, nofact_weight=None, num_tokens=None, clf_loss=None):
        super().__init__()
        assert True if nofact_weight is None else (False if num_tokens is None else num_tokens > nofact_token_id), \
            f"Invalid combination of nofact_weight '{nofact_weight}', num_tokens '{num_tokens}' and nofact_token_id '{nofact_token_id}'"
        assert True if clf_loss is None else (clf_loss == 'reweighted' or clf_loss == 'inverse'), f"Invalid value for clf_loss '{clf_loss}'"
        self.nofact_token_id = nofact_token_id
        self.clf_loss = clf_loss
        self.ignore_index = ignore_index
        if nofact_weight is not None:
            weight = torch.ones(num_tokens)
            weight[nofact_token_id] = nofact_weight
            self.nllloss = nn.NLLLoss(ignore_index=ignore_index, weight=weight, reduction='none')
        else:
            self.nllloss = nn.NLLLoss(ignore_index=ignore_index, reduction='none')
        self.lm_weight = lm_weight

    def fact_loss(self, lm_logprobs, target):
        if self.clf_loss is None:
            fact_loss = self.nllloss(lm_logprobs, target)
        else:
            logprob_nofact = lm_logprobs[:, self.nofact_token_id]
            all_tokens_ids = torch.arange(lm_logprobs.shape[1])
            all_tokens_ids_except_nofact = all_tokens_ids[all_tokens_ids != self.nofact_token_id]
            maxlogprob_fact = lm_logprobs[:, all_tokens_ids_except_nofact].max(dim=1)[0]
            if self.clf_loss == 'reweighted':
                fact_loss = torch.where(target == self.nofact_token_id, -logprob_nofact, -maxlogprob_fact)  # This is old version
            elif self.clf_loss == 'inverse':
                fact_loss = torch.where(target == self.nofact_token_id, -1/maxlogprob_fact, -1/logprob_nofact)
        return fact_loss

    def forward(self, lm_logprobs, target, seq_start=1):
        # Use seq_start=1 if sequence starts with <bos> token, otherwise use seq_start=0

        assert (lm_logprobs.shape[2] > seq_start) and (target.shape[1] > seq_start), f"Invalid shape for lm_logprobs {input.shape} or target {target.shape}"

        # Classification loss: whether facts are recognized correctly
        # Token directly after <bos> token signals whether sequence is a fact
        classification_loss = self.fact_loss(lm_logprobs[:, :, seq_start], target[:, seq_start])

        # LM loss: whether the tokens of the facts are predicted correctly; exclude <bos> from loss calculation
        lm_loss = self.nllloss(lm_logprobs[:, :, seq_start:], target[:, seq_start:]).sum(dim=1) / (target[:, seq_start:] != self.ignore_index).sum(dim=1)

        # Weighted combination of classification loss and LM loss
        combined_loss =  (1 - self.lm_weight) * classification_loss + self.lm_weight * lm_loss
        
        return combined_loss.mean(), classification_loss.mean(), lm_loss.mean()

class BartExtractor(nn.Module):

    batch_format = "huggingface"

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('BartExtractor')
        group.add_argument("--lm_loss_factor", type=float, default=1.0, help="relative weight of lm_loss in combined loss")
        group.add_argument("--nofact_weight", type=float, default=None, help="weight for nofact token in loss function")
        group.add_argument("--clf_loss", default=None, choices=['reweighted', 'inverse'], help="variant for classification loss")
        group.add_argument("--decoder_max", type=int, default=50, help="max number of tokens to generate")
        group.add_argument("--bart_base", type=str, default='facebook/bart-large-cnn', help="name of pretrained BART model")
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

    def forward(self, input_ids, attention_mask, labels):
        # NOTE: within bart.forward(), a decoder_start_token </s> is inserted (--> label shifted right)
        output = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_logits = output.logits
        lm_logprobs = self.logsoftmax(lm_logits)
        return lm_logprobs
        
    def _update_generation_args(self, **generation_args):
        """
        Override the standard generation args for bart model:
        - set minimum length to 2
        - set conditional generation, to ensure <eos> is generated after <nofact>
        """
        def allowed(batch_id, sent):
            if sent[-1] == self.nofact_token_id:
                return [self.bart.config.eos_token_id, self.bart.config.pad_token_id] # NOTE: pad token added here, to prevent crash n beamsample
            else:
                return list(range(self.bart.config.vocab_size))
        
        generation_args['min_length'] = 2
        generation_args['prefix_allowed_tokens_fn'] = allowed
        
        return generation_args
    
    def fact_mask(self, output_ids):
        mask = output_ids != self.nofact_token_id
        mask *= output_ids !=self.bart.config.bos_token_id
        mask *= output_ids !=self.bart.config.eos_token_id
        mask *= output_ids !=self.bart.config.pad_token_id
        return mask

    def generate(self, input_ids, **kwargs):       
        """
        NOTE: Within generate, first the input is encoded; attention_mask is defined by filtering on token != pad_token
        A decoder_start_token is generated as first token of the output
        """
        kwargs = self._update_generation_args(**kwargs)
        gen_out = self.bart.generate(input_ids, **kwargs)
        pred_fact = torch.any(self.fact_mask(gen_out), dim=1)
        # pred_fact = gen_out[:, 2] != self.nofact_token_id

        logging.spam("Generate: pred_fact={}".format(pred_fact))
        logging.spam("Generate: gen_out={}".format(gen_out))

        return gen_out

    def train_step(self, batch, optimizer, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)

        optimizer.zero_grad()
        lm_logprobs = self.forward(input_ids, attention_mask, y)
        loss, classification_loss, lm_loss = criterion(lm_logprobs.transpose(1,2), y)
        logging.debug(f"Train: loss {loss:.4f}, clf_loss {classification_loss:.4f}, lm_loss {lm_loss:.4f}")
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def valid_step(self, batch, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)

        with torch.no_grad():
            lm_logprobs = self.forward(input_ids, attention_mask, y)
            loss, classification_loss, lm_loss = criterion(lm_logprobs.transpose(1,2), y)

        pred = lm_logprobs.cpu().argmax(dim=-1)
        ignore_mask = batch['labels'].ne(self.bart.config.pad_token_id)
        
        # Classification accuracy
        pred_fact = pred[:, 1] != self.nofact_token_id  # Check for nofact-token, directly after the start-of-sentence
        label_fact = batch['labels'][:, 1] != self.nofact_token_id # In label, nofact-token would be at position 1, after the <bos>-token
        fact_correct = label_fact.eq(pred_fact)
        fact_acc = fact_correct.sum().item() / batch['labels'].shape[0]

        # LM perplexity (only in case the target has a fact; calculated excluding the <bos>-token)
        target_fact = y[:,1] != self.nofact_token_id
        ppl = perplexity(preds=lm_logprobs[target_fact, 1:, :], target=y[target_fact, 1:], ignore_index=criterion.ignore_index).item()

        # LM accuracy (only in case the target has a fact; calculated excluding the <bos>-token)
        target_fact = target_fact.cpu()
        token_correct = batch['labels'].eq(pred) * ignore_mask
        token_acc = (token_correct[target_fact, 1:].sum() / ignore_mask[target_fact, 1:].sum()).item() 

        stats = {
            "loss": loss.item(),
            "classification_loss": classification_loss.item(),
            "lm_loss": lm_loss.item(),
            "acc": fact_acc,
            "perplexity": ppl, 
            "token_prediction_acc": token_acc
        }
        logging.debug(f"Valid: loss {loss:.4f}, clf_loss {classification_loss:.4f}, lm_loss {lm_loss:.4f}, cls_acc {fact_acc:.4f}, lm_acc {token_acc:.4f}, ppl {ppl:.4f}")

        return stats

##
## The following is not used
## PrefixBart model should be fixed and tested
##

class PrefixBart(BartExtractor):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('PrefixBart')
        group.add_argument("--freeze", type=int, default=0, help="Layers to freeze for finetuning; None=none, 0=only embeddings, 12=all")
        group.add_argument("--enc_prefix_size", type=int, default=0, help="Insert prefix in BART encoder")
        group.add_argument("--dec_prefix_size", type=int, default=0, help="Insert prefix in BART decoder")
        group.add_argument("--lm_loss_factor", type=float, default=0.5, help="Relative weight of lm_loss in combined loss")
        group.add_argument("--prefix_aggr", type=str, default="concat", choices=["concat", "max", "avg"], help="How to aggregate prefix hidden states")
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
        self.nofact_sequence = torch.tensor([self.bart.config.decoder_start_token_id, self.bart.config.bos_token_id, self.nofact_token_id, self.bart.config.eos_token_id])
        if self.nofact_token_id == self.bart.config.eos_token_id:
            self.nofact_sequence = self.nofact_sequence[:-1]
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
    import argparse
    import random
    from transformers import AutoTokenizer
    from dataset.msc_summary_turns import MSC_Turns

    parser = argparse.ArgumentParser(description="Test MSC_Summary", conflict_handler="resolve")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--log_interval", type=int, default=10, help="report interval")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())    
    parser.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    parser.add_argument("--load", type=str, default="", help="filename of model to load")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--datadir", type=str, default="./data/", help="Datadir")
    parser.add_argument("--basedir", type=str, default="msc/msc_personasummary/", help="Base directory for dataset")
    parser.add_argument("--savedir", type=str, default="./output/", help="directory for output files")

    parser = MSC_Turns.add_cmdline_args(parser)
    parser = BartExtractor.add_cmdline_args(parser)
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare logging
    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    # Settings for test
    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/'
    args.sessions = [1]
    subset = 'train'
    args.device="cpu"
    args.load="trained_bart"

    args.speaker_prefixes = ["<you>", "<me>"]
    args.nofact_token = '<nofact>'
    args.add_tokens = args.speaker_prefixes + [args.nofact_token]
    model_class = "bart"
    args.lm_loss_factor = 0.5
    # args.freeze=None
    # args.enc_prefix_size=0
    # args.dec_prefix_size=0
    # args.prefix_aggr="concat"

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(args.bart_base)
    if args.add_tokens is not None:
        num_added_toks = tokenizer.add_tokens(args.add_tokens)
    nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
    assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

    if model_class == "bart":
        model = BartExtractor(bart_base=args.bart_base, nofact_token_id=nofact_token_id)
    else:
        model = PrefixBart(
            bart_base=args.bart_base, 
            nofact_token_id=nofact_token_id, 
            freeze=args.freeze, 
            enc_prefix_size=args.enc_prefix_size,
            dec_prefix_size=args.dec_prefix_size,
            prefix_aggr=args.prefix_aggr
        )
    model.bart.resize_token_embeddings(len(tokenizer))
    criterion = ExtractedFactLoss(nofact_token_id=nofact_token_id, ignore_index=-100, lm_weight=args.lm_loss_factor)

    MSC_Turns.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
    msc_turns = MSC_Turns(basedir=basedir, sessions=args.sessions, subset=subset)
    data = [msc_turns[i] for i in range(10)]
    batch = msc_turns.batchify(data, batch_format=model.batch_format)

    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        logging.info("Loading model from {}".format(loadpath))
        model.load_state_dict(torch.load(loadpath, map_location=torch.device(args.device)))

    # Test forward
    lm_logprobs = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
    preds = lm_logprobs.argmax(dim=-1)
    response = tokenizer.batch_decode(preds)

    for i in range(3):
        print('-' * 40)
        print(data[i][0])
        print(data[i][1])
        print(response[i])

    # Test loss function and valid_step
    valid_stats = model.valid_step(batch, criterion, device=args.device)
    logging.report("valid_stats = {}".format(valid_stats))

    # Test generate
    with torch.no_grad():
        # pred_tokens = model.generate(batch['input_ids'], max_new_tokens=20)
        pred_tokens = model.generate(
            batch['input_ids'].to(args.device), 
            # min_length=2,
            max_new_tokens=20, 
            # num_beams=1,
            # do_sample=False,
        )
    pred_persona = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("GENERATE")
    for i, (t, p) in enumerate(zip(pred_tokens, pred_persona)):
        print('-' * 40)
        print('context:    ', data[i][0])
        print('target:     ', data[i][1])
        print('tokens:     ', t)
        print('prediction: ', p)




