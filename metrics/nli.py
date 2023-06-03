###
### Functions to calculate use NLI as metric to assess quality of facts that are generated based on history of dialogue utterances
###

from torchmetrics import Metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NLIMetric(Metric):

    nli_model = None
    tokenizer = None
    device = None
    batch_size = None

    @classmethod
    def set(cls, nli_model, device, batch_size):
        cls.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        cls.tokenizer = AutoTokenizer.from_pretrained(nli_model)
        cls.device = device
        cls.nli_model.to(device)
        cls.batch_size = batch_size

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('NLIMetric')
        group.add_argument("--nli_model", type=str, default='facebook/bart-large-mnli', help="model to use for NLI metric")
        return parser

    def __init__(self):
        super().__init__()
        self.add_state("info", default=[], dist_reduce_fx="cat", persistent=False)
        self.add_state("turns", default=[], dist_reduce_fx="cat", persistent=False)
        self.add_state("predictions", default=[], dist_reduce_fx="cat", persistent=False)

    def update(self, id, turn, pred):
        self.info.append(id)
        self.turns.append(turn)
        self.predictions.append(pred)

    def compute(self):
        
        # Check if metric has been configured correctly
        assert self.nli_model is not None
        assert self.tokenizer is not None

        stats = {}

        for i in range(0, len(self.turns), self.batch_size):

            info = [self.info[i+j] for j in range(self.batch_size) if i + j < len(self.info)]
            turns = [self.turns[i+j] for j in range(self.batch_size) if i + j < len(self.info)]
            preds = [self.predictions[i+j] for j in range(self.batch_size) if i + j < len(self.info)]

            # Use NLI model to assess whether predictions can be infered from utterances
            encoded = self.tokenizer(text=turns, text_pair=preds, return_tensors='pt', padding=True, truncation='only_first')
            encoded = encoded.to(self.device)
            output = self.nli_model(**encoded)

            # Collect the statistics
            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true 
            entail_contradiction_logits = output.logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            stats.update({inf: prob[1].item() for inf, prob in zip(info, probs)})

        return stats

if __name__ == '__main__':
    from dataset.msc_summary_turns import MSC_Turns

    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/'
    sessions = [2]
    subset = 'train'
    speaker_prefixes = ["You: ", "Me: "]
    nofact_token = ''
    add_tokens = speaker_prefixes + [nofact_token]

    # Test extraction of dialogue turns and persona sentences
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    if add_tokens is not None:
        tokenizer.add_tokens(add_tokens)

    MSC_Turns.set(tokenizer=tokenizer, len_context=2, speaker_prefixes=speaker_prefixes, nofact_token=nofact_token)

    msc_turns = MSC_Turns(
        basedir=basedir, 
        sessions=sessions, 
        subset=subset,
        max_samples=20
    )

    NLIMetric.set(nli_model='facebook/bart-large-mnli', device='cpu')
    nli_metric = NLIMetric()
    for i in range(len(msc_turns)):
        nli_metric.update(msc_turns.indices[i]['convai_id'], msc_turns[i][0], msc_turns[i][1] if msc_turns[i][1] is not None else "")
    stats = nli_metric.compute()

    for i,s in enumerate(stats.values()):
        if msc_turns[i][1] != nofact_token:
            print("Id      : ", msc_turns.indices[i]['convai_id'])
            print("History : ", msc_turns[i][0])
            print("Target  : ", msc_turns[i][1])
            print(f"Score   :  {s:.4f}")
            print('-' * 40)