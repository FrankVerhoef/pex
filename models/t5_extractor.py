import torch
import torch.nn as nn

import utils.logging as logging

from transformers import T5ForConditionalGeneration, T5Config
from torchmetrics.functional.text.perplexity import perplexity


class T5Extractor(nn.Module):

    batch_format = "huggingface_t5"

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('T5Extractor')
        group.add_argument("--lm_loss_factor", type=float, default=1.0, help="relative weight of lm_loss in combined loss")
        group.add_argument("--nofact_weight", type=float, default=None, help="weight for nofact token in loss function")
        group.add_argument("--clf_loss", default=None, choices=['reweighted', 'inverse'], help="variant for classification loss")
        group.add_argument("--decoder_max", type=int, default=50, help="max number of tokens to generate")
        group.add_argument("--t5_base", type=str, default='t5-small', help="name of pretrained T5 model")
        return parser

    def __init__(self, t5_base=None, nofact_token_id=None):
        super().__init__()
        assert nofact_token_id is not None, "Must provide nofact_token_id"
        self.nofact_token_id = nofact_token_id
        if t5_base is None:
            self.t5 = T5ForConditionalGeneration(config=T5Config())
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained(t5_base)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels):
        # NOTE: within t5.forward(), a decoder_start_token (which is the padding token) is inserted (--> label shifted right)
        output = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_logits = output.logits
        lm_logprobs = self.logsoftmax(lm_logits)
        return lm_logprobs
        
    def _update_generation_args(self, **generation_args):
        """
        Override the standard generation args for t5 model:
        - set minimum length to 2
        - set conditional generation, to ensure <eos> is generated after <nofact>
        """
        def allowed(batch_id, sent):
            if sent[-1] == self.nofact_token_id:
                return [self.t5.config.eos_token_id, self.t5.config.pad_token_id] # NOTE: pad token added here, to prevent crash n beamsample
            else:
                return list(range(self.t5.config.vocab_size))
        
        generation_args['min_length'] = 2
        generation_args['prefix_allowed_tokens_fn'] = allowed
        
        return generation_args
    
    def fact_mask(self, output_ids):
        # If the output contains any tokens that is not <nofact>, <bos>, <eos> or <pad>, then consider it a fact
        mask = output_ids != self.nofact_token_id
        mask *= output_ids !=self.t5.config.bos_token_id
        mask *= output_ids !=self.t5.config.eos_token_id
        mask *= output_ids !=self.t5.config.pad_token_id
        return mask

    def generate(self, input_ids, **kwargs):       
        """
        NOTE: Within generate, first the input is encoded; attention_mask is defined by filtering on token != pad_token
        A decoder_start_token is generated as first token of the output
        """
        kwargs = self._update_generation_args(**kwargs)
        gen_out = self.t5.generate(input_ids, **kwargs)
        pred_fact = torch.any(self.fact_mask(gen_out), dim=1)

        logging.spam("Generate: pred_fact={}".format(pred_fact))
        logging.spam("Generate: gen_out={}".format(gen_out))

        return gen_out

    def train_step(self, batch, optimizer, criterion, device):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)

        optimizer.zero_grad()
        lm_logprobs = self.forward(input_ids, attention_mask, y)
        loss, classification_loss, lm_loss = criterion(lm_logprobs.transpose(1,2), y, seq_start=0)
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
            loss, classification_loss, lm_loss = criterion(lm_logprobs.transpose(1,2), y, seq_start=0)

        pred = lm_logprobs.cpu().argmax(dim=-1)
        ignore_mask = batch['labels'].ne(self.t5.config.pad_token_id)
        
        # Classification accuracy
        pred_fact = pred[:, 0] != self.nofact_token_id  # Check for nofact-token, directly at the start-of-sentence
        label_fact = batch['labels'][:, 0] != self.nofact_token_id # In label, nofact-token would be at position 0
        fact_correct = label_fact.eq(pred_fact)
        fact_acc = fact_correct.sum().item() / batch['labels'].shape[0]

        # LM perplexity (only in case the target has a fact)
        target_fact = y[:,0] != self.nofact_token_id
        ppl = perplexity(preds=lm_logprobs[target_fact, :, :], target=y[target_fact, :], ignore_index=criterion.ignore_index).item()

        # LM accuracy (only in case the target has a fact)
        target_fact = target_fact.cpu()
        token_correct = batch['labels'].eq(pred) * ignore_mask
        token_acc = (token_correct[target_fact, :].sum() / ignore_mask[target_fact, :].sum()).item() 

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


if __name__ == "__main__":
    import argparse
    import random
    from transformers import AutoTokenizer
    from dataset.msc_summary_turns import MSC_Turns
    from models.bart_extractor import ExtractedFactLoss

    parser = argparse.ArgumentParser(description="Test MSC_Summary with T5", conflict_handler="resolve")
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
    parser = T5Extractor.add_cmdline_args(parser)
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
    # args.load="trained_bart"

    args.speaker_prefixes = ["<you>", "<me>"]
    args.nofact_token = '<nofact>'
    args.add_tokens = args.speaker_prefixes + [args.nofact_token]
    model_class = "t5"
    args.lm_loss_factor = 0.5

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(args.t5_base)
    if args.add_tokens is not None:
        num_added_toks = tokenizer.add_tokens(args.add_tokens)
    nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
    assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

    if model_class == "t5":
        model = T5Extractor(t5_base=args.t5_base, nofact_token_id=nofact_token_id)
    model.t5.resize_token_embeddings(len(tokenizer))
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




