import torch

import utils.logging as logging

from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from torchmetrics.functional.text.perplexity import perplexity

class DialoGPT(PreTrainedModel):

    batch_format = "huggingface_xycat"

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('DialoGPT')
        group.add_argument("--lm", type=str, default="microsoft/DialoGPT-small", help="Name of language model")
        group.add_argument("--decoder_max", type=int, default=50, help="Max number of tokens to generate")
        return parser

    def __init__(self, lm, bos_token_id):
        super().__init__(config=PretrainedConfig())
        self.model = AutoModelForCausalLM.from_pretrained(lm)
        self.bos_token_id = bos_token_id

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def train_step(self, batch, optimizer, criterion, device):

        inputs = batch.to(device)

        optimizer.zero_grad()
        output = self.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        loss = criterion(output.logits[:, :-1].transpose(1,2), inputs.input_ids[:, 1:])
        logging.debug("Train: loss {:.4f}".format(loss.item()))
        loss.backward()
        optimizer.step()
        
        return loss.item()
    

    def valid_step(self, batch, criterion, device):

        inputs = batch.to(device)
    
        with torch.no_grad():
            output = self.forward(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            loss = criterion(output.logits[:, :-1].transpose(1,2), inputs.input_ids[:, 1:])

        logits = output.logits[:, :-1, :].to("cpu")
        pred = logits.argmax(dim=-1)
        inputs = inputs.to("cpu")

        # LM accuracy
        token_correct = inputs.input_ids[:, 1:].eq(pred) * inputs.attention_mask[:, 1:]
        token_acc = (token_correct.sum() / inputs.attention_mask[:, 1:].sum()).item() 

        # LM perplexity
        ppl = perplexity(preds=logits, target=inputs.input_ids[:, 1:], ignore_index=self.model.config.pad_token_id).item()

        stats = {
            "loss": loss.mean().item(),
            "acc": token_acc,
            "perplexity": ppl
        }

        logging.debug("Valid: loss {:.4f}, ppl {:.2f}".format(loss.mean().item(), ppl))
        return stats


if __name__ == "__main__":

    import argparse
    import random
    import torch.nn as nn
    from torch import optim
    from transformers import AutoTokenizer, PretrainedConfig, GenerationConfig
    from dataset.msc_sessions import MSC_Session
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test DialoGPT")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--valid_interval", type=int, default=None, help="validation interval")
    parser.add_argument("--patience", type=int, default=None, help="Number of validation intervals without improvement after which training will be terminated")

    parser = DialoGPT.add_cmdline_args(parser)
    parser = MSC_Session.add_cmdline_args(parser)

    args = parser.parse_args()
    print(vars(args))

    def predict(obs_batch, model, tokenizer, device, collate_fn):
        inputs = collate_fn(obs_batch, with_labels=False, batch_format=model.batch_format, buffer=25)
        B, L = inputs.input_ids.shape[:2]
        bos_tokens = torch.full((B, 1), fill_value=model.bos_token_id, dtype=torch.long, device=inputs.input_ids.device)
        model.to(device)

        with torch.no_grad():
            output = model.model.generate(
                # Add the bos_token to the input. 
                inputs=torch.cat([inputs.input_ids, bos_tokens], dim=1).to(device), 
                generation_config=GenerationConfig(
                    pad_token_id=model.model.config.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=25,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            )
        output_gen = output.sequences[:, L+1:]
        logging.verbose(output_gen)
        responses = tokenizer.batch_decode(output_gen)
        return responses

    logging.set_log_level(logging.SPAM)
    # logging.set_only_message(True)

    # Settings for test
    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    args.speaker_prefixes = ['<me>', '<you>']
    args.add_tokens = None #args.speaker_prefixes
    if args.add_tokens is not None:
        tokenizer.add_tokens(args.add_tokens)
    tokenizer.bos_token_id = tokenizer.eos_token_id
    if args.speaker_prefixes is not None:
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(args.speaker_prefixes[0])

    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_dialogue/'
    MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=args.speaker_prefixes)
    dataset = MSC_Session(
        basedir=basedir,
        session=2,
        subset='test', 
        include_persona=True,
        include_history=True,
        augmented=True,
        max_samples=3, 
    )
    model = DialoGPT(args.lm, tokenizer.bos_token_id)
    model.model.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    data = [dataset[i] for i in range(args.batch_size)]
    batch = dataset.batchify(data, with_labels=False, batch_format=model.batch_format)

    responses = predict(data, model, tokenizer, device=args.device, collate_fn=MSC_Session.batchify)
    responses_stringlist = [
        "Context:  {}\n"
        "Label:    {}\n"
        "Response: {}\n"
        "{}\n".format(text, label, response, "-" * 20)
        for (text, label), response in zip(data, responses)
    ]
    logging.success("Generate output:\n{}".format('\n'.join(responses_stringlist)))

    model.to(args.device)
    output = model.train_step(batch, optimizer, criterion, args.device)
    logging.report("Train_step output: {}".format(output))

    output = model.valid_step(batch, criterion, args.device)
    logging.report("Valid_step output: {}".format(output))
