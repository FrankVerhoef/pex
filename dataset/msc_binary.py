from torch.utils.data import Dataset
from dataset.msc_summary import MSC_Turns, persona_token
import torch


class MSC_Turn_Facts(Dataset):

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('MSC_Turn_Facts')
        group.add_argument("--persona_tokens", type=bool, default=False, help="Whether to insert special persona token before each dialogue turn")
        return parser

    def __init__(self, path, tokenizer, len_context=2, persona_tokens=False, max_samples=None):
        """
        Variant of the MSC dataset that can be used to learn whether a series of utterances implies a new fact about the last speaker.
        Builds on the MSC_Turns dataset
        """

        super(MSC_Turn_Facts, self).__init__()
        self.tokenizer = tokenizer
        self.turns = MSC_Turns(path, None, len_context, persona_tokens, max_samples)
        self.has_fact = [persona != '' for turn, persona in self.turns]

    def __len__(self):
        return len(self.turns)
    
    def __getitem__(self, index):
        """
        Returns two sentences and boolean indicating whether the sentences imply a new fact
        First sentence is composed of all utterances in the context, except the last (so maximum of 'len_context' - 1 utterances);
        Second sentence is last utterance in context
        """
        turn = self.turns.get_turn(index)
        if self.turns.persona_tokens:
            turn_0 = ' '.join([persona_token[p] + ' ' + t for p, t in turn[:-1]])
            turn_1 = persona_token[turn[-1][0]] + ' ' + turn[-1][1]
        else:
            turn_0 = ' '.join([t for _, t in turn[:-1]])
            turn_1 = turn[-1][1]
        return turn_0, turn_1, self.has_fact[index]

    def batchify(self, data):
        """
        Collate all items into batch for classification model.
        This function assumes the tokenizer is a Huggingface tokenizer, that takes two batches of input sentences and
        returns input_ids, attention_mask and token_type_ids.
        Returns a tuple a tuple with (input_ids, attention_mask, token_type_ids) and labels
        """
        turns_0, turns_1, has_fact = zip(*data)
        encoded = self.tokenizer(turns_0, turns_1, padding=True, return_tensors='pt')
        X = (encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids'])
        y = torch.tensor(has_fact, dtype=torch.long) #.reshape(-1, 1)

        return X, y

if __name__ == "__main__":

    from transformers import AutoTokenizer
    from dataset.msc_summary import PERSONA_TOKENS

    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/session_1/train.txt'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    persona_tokens = False
    if persona_tokens:
        num_added_toks = tokenizer.add_tokens(PERSONA_TOKENS)
    
    # Test extraction of dialogue turns and persona facts
    msc_turns = MSC_Turn_Facts(datapath, tokenizer=tokenizer, len_context=4, persona_tokens=persona_tokens)

    batch = [msc_turns[i] for i in range(10)]

    for item in batch:
        print(item[0])
        print(item[1])
        print(item[2])
        print('-'*40)

    print(msc_turns.batchify(batch))
    print('-' * 40)
