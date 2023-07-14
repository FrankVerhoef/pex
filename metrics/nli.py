###
### Functions to calculate use NLI as metric to assess quality of facts that are generated based on history of dialogue utterances
###

import torch
from torchmetrics import Metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import utils.logging as logging

CONTRADICTION_THRESHOLD = 0.5
ENTAILMENT_THRESHOLD = 0.5

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
            entail_contradiction_logits = output.logits 
            probs = entail_contradiction_logits.softmax(dim=1)
            stats.update({inf: prob[2].item() for inf, prob in zip(info, probs)})

        return stats

class ConsistencyMetric(NLIMetric):
    """
    Calculate consistency score between the prediction and all the previous utterances of the speaker
    """

    def compute(self, nli_type='ENTAILMENT'):
        
        # Check if metric has been configured correctly
        assert self.nli_model is not None
        assert self.tokenizer is not None
        assert nli_type in ['CONTRADICTION', 'ENTAILMENT']

        stats = {}
        nli_index, threshold = {
            'ENTAILMENT': (2, ENTAILMENT_THRESHOLD),
            'CONTRADICTION': (0, CONTRADICTION_THRESHOLD)
        }[nli_type]

        for i in range(len(self.info)):

            nli_probs = []
            for u_id in range(0, len(self.turns[i]), self.batch_size):
                utterances = [self.turns[i][u_id + j] for j in range(self.batch_size) if u_id + j < len(self.turns[i])]
                preds = [self.predictions[i]] * len(utterances)

                # Use NLI model to assess whether predictions are consistent with utterances
                with torch.no_grad():
                    encoded = self.tokenizer(text=utterances, text_pair=preds, return_tensors='pt', padding=True, truncation='only_first')
                    encoded = encoded.to(self.device)
                    output = self.nli_model(**encoded)

                # Collect the statistics
                entail_contradiction_logits = output.logits 
                probs = entail_contradiction_logits.softmax(dim=1)
                nli_probs.append(probs)

            scores = torch.cat(nli_probs, dim=0)[:,nli_index].to("cpu") # extract the required nli_probs
            stats[self.info[i]] = {
                nli_type: torch.any(scores > threshold).item(),
                'nli_premises' : [u for u, s in zip(self.turns[i], scores) if s > threshold]
            }
            logging.spam(f"Prediction: {self.predictions[i]}\n" \
                + "\n".join([f"{'*' if s > threshold else ' '} {s:.2f} {u}" for u, s in zip(self.turns[i], scores)]) \
                + '\n-------\n')

        return stats 

if __name__ == '__main__':
    from dataset.msc_summary_turns import MSC_Turns
    from dataset.msc_sessions import MSC_Session

    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

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
        basedir='/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/', 
        sessions=sessions, 
        subset=subset,
        max_samples=20
    )

    # NLIMetric.set(nli_model='facebook/bart-large-mnli', device='cpu', batch_size=8)
    # nli_metric = NLIMetric()
    # for i in range(len(msc_turns)):
    #     nli_metric.update(msc_turns.indices[i]['convai_id'], msc_turns[i][0], msc_turns[i][1] if msc_turns[i][1] is not None else "")
    # stats = nli_metric.compute()

    # for i,s in enumerate(stats.values()):
    #     if msc_turns[i][1] != nofact_token:
    #         print("Id      : ", msc_turns.indices[i]['convai_id'])
    #         print("History : ", msc_turns[i][0])
    #         print("Target  : ", msc_turns[i][1])
    #         print(f"Score   :  {s:.4f}")
    #         print('-' * 40)

    MSC_Session.set(sessionbreak_token='<session>')
    msc_sessions = MSC_Session(
        basedir='/Users/FrankVerhoef/Programming/PEX/data/msc/msc_dialogue/', 
        session=1, 
        subset=subset,
        include_persona=True,
        include_history=True,
        max_samples=20
    )

    ConsistencyMetric.set(nli_model='facebook/bart-large-mnli', device='cpu', batch_size=8)
    cons_metric = ConsistencyMetric()
    for i in range(len(msc_sessions)):
        cons_metric.update(
            (msc_sessions.indices[i]["dialog_id"], msc_sessions.indices[i]["turn_id"]), 
            msc_sessions.personas(i, msc_sessions.next_utterance[i][0]),
            msc_sessions.next_utterance[i][1]
        )
    cons_stats = cons_metric.compute('ENTAILMENT')
    for i,s in enumerate(cons_stats.values()):
        print("Id             : ", msc_sessions.indices[i])
        print("Next utterance : ", msc_sessions[i][1])
        print(f"Score          :  {s['ENTAILMENT']}")
        print("Premises       : ", s["nli_premises"])
        print('-' * 40)