import torch
import torch.nn as nn

import utils.logging as logging

from models.encoder_models import ENCODERS, ENCODER_TYPES
from models.decoder_models import DECODERS, DECODER_TYPES


class PersonaExtractor(nn.Module):

    batch_format = "padded_sequences"

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('PersonaExtractor args')
        group.add_argument("--encoder", type=str, default="mean", help="Encoder model")
        group.add_argument("--embedding_size", type=int, default=100, help="Embedding size")
        group.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
        group.add_argument("--aggregate_method", type=str, default="avg", choices=["avg", "max"], help="Aggregate method for Pool Bi-LSTM")
        group.add_argument("--decoder", type=str, default="lstm", help="Decoder model")
        group.add_argument("--decoder_max", type=int, default=20, help="Max number of tokens to generate with decoder")
        return parser

    def __init__(self, encoder_type, encoder_opts, decoder_type, decoder_opts, start_token, nofact_token_id):
        super().__init__()
        self.embed = nn.Embedding(encoder_opts["input_size"], encoder_opts["embedding_size"])
        self.encoder = ENCODERS[encoder_type](encoder_opts)
        self.decoder = DECODERS[decoder_type](decoder_opts)
        self.start_token = start_token
        self.nofact_token_id = nofact_token_id


    def forward(self, xs, xs_len, max=20, teacher_forcing=False, labels=None):

        # Check validity of input
        if teacher_forcing:
            assert labels is not None, "Need to supply target sequence for teacher forcing"
            max = labels.size(1)
        
        # Apply encoder to input sequences
        embeds = self.embed(xs)
        encoded = self.encoder(embeds, xs_len)

        # Set initial state for decoder
        B, H = encoded.shape
        decoder_input = torch.full(size=(B, 1), fill_value=self.start_token, device=encoded.device)
        state = (encoded.unsqueeze(dim=0), torch.zeros((1, B, H), device=encoded.device))

        # Generate tokens based on output of encoder
        output = []
        for i in range(max):
            embeds = self.embed(decoder_input)
            out, state = self.decoder(embeds, state)
            output.append(out)
            if teacher_forcing:
                decoder_input = labels[:, i].view(B, 1)
            else:
                decoder_input = out.argmax(dim=-1)

        return torch.stack(output, dim=-1).reshape(B, len(output), -1)
  
    def generate(self, xs, xs_len, max=20):
        out = self.forward(xs, xs_len, max=max, teacher_forcing=False, labels=None)
        gen_out = out.argmax(dim=-1)
        return gen_out

    def train_step(self, batch, optimizer, criterion, device):

        xs, ys, xs_len, ys_len = batch
        xs = xs.to(device)
        ys = ys.to(device)
        optimizer.zero_grad()
        
        output = self.forward(xs, xs_len, teacher_forcing=True, labels=ys)
        loss = criterion(output.transpose(1,2), ys)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    

    def valid_step(self, batch, criterion, device):

        xs, ys, xs_len, ys_len = batch
        xs = xs.to(device)
        ys = ys.to(device)

        with torch.no_grad():
            output = self.forward(xs, xs_len, max=ys.size(1), teacher_forcing=True, labels=ys)
            loss = criterion(output.transpose(1,2), ys)

        pred = output.cpu().argmax(dim=-1)

        ys = ys.cpu()
        ignore_mask = ys.ne(criterion.ignore_index)

        # Classification accuracy
        pred_fact = pred[:, 0] != self.nofact_token_id  # Check for nofact-token, directly at the start-of-sentence
        label_fact = ys[:, 0] != self.nofact_token_id
        fact_correct = label_fact.eq(pred_fact)
        fact_acc = fact_correct.sum().item() / ys.shape[0]

        # LM accuracy
        token_correct = ys.eq(pred) * ignore_mask
        token_acc = (token_correct.sum() / ignore_mask.sum()).item() 

        stats = {
            "loss": loss.item(),
            # "classification_loss": classification_loss.item(),
            # "lm_loss": loss.item(),
            "acc": fact_acc,
            "token_pred?iction_acc": token_acc
        }
        logging.debug("Valid: loss {:.4f}, cls_acc {:.4f}, lm_acc {:.4f}".format(loss, fact_acc, token_acc))

        return stats

if __name__ == '__main__':
    from torch import optim
    import copy
    import random

    I = 60
    L = 20
    E = 16
    H = 48
    B = 8

    def print_params(model):
        print("Model parameters")
        for p in model.parameters():
            print("\t{:<30} {}".format(str(p.data.shape), p.data.norm()))

    def print_grads(model):
        # print("Gradients")
        sum_grads = 0
        for p in model.parameters():
            # print("\t{:<30} {}".format(str(p.grad.shape), p.grad.norm()))
            sum_grads += p.grad.norm()
        return sum_grads


    encoder_opts = {
        "input_size": I,
        "embedding_size": E,
        "hidden_size": H,
        "aggregate_method": "max"
    }

    random.seed(42)
    torch.manual_seed(42)

    criterion = nn.NLLLoss()

    X = torch.randint(high=I, size=(B, L))
    X_lens = torch.randint(low=1, high=L, size=(B, ))
    y = torch.randint(high=I, size=(B, L))

    results = []

    for encoder_type in ENCODER_TYPES:
        for decoder_type in DECODER_TYPES:

            decoder_opts = {
                "input_size": I,
                "embedding_size": E,
                "hidden_size": {
                    "mean": E,
                    "lstm": H,
                    "bilstm": H * 2,
                    "poolbilstm": H * 2            
                }[encoder_type],
                "output_size": I
            }
            basemodel = PersonaExtractor(encoder_type, encoder_opts, decoder_type, decoder_opts, start_token=1, nofact_token_id=2)

            for dev in ["cpu", "mps"]:
                model = copy.deepcopy(basemodel).to(dev)
                # print_params(model)

                optimizer = optim.Adam(model.parameters(), lr=0.001)

                Xd = X.to(dev)
                X_lens = torch.randint(low=1, high=L, size=(B, ))
                yd = y.to(dev)

                optimizer.zero_grad()
                out = model(Xd, X_lens, teacher_forcing=True, labels=yd)
                loss = criterion(out.transpose(1,2), yd)
                # print("Loss: ", loss)

                loss.backward()
                all_grads = print_grads(model)

                results.append({'dev': dev, 'enc': encoder_type, 'dec': decoder_type, 'loss': loss, 'grads': all_grads})

    print('-' * 43)
    print("{:<4} {:<11} {:<4} {:>10} {:>10}".format('dev', 'encoder', 'dec', 'loss', 'grads'))
    print('-' * 43)
    for r in results:
        print("{:<4} {:<11} {:<4} {:10.4f} {:10.4f}".format(r['dev'], r['enc'], r['dec'], r['loss'], r['grads']))         

