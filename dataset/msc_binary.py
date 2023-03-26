from torch.utils.data import Dataset
from dataset.msc_summary import MSC_Turns
import torch
from torcheval.metrics.functional import binary_confusion_matrix, binary_accuracy, binary_f1_score

class MSC_Turn_Facts(MSC_Turns):

    """
    Variant of the MSC dataset that can be used to learn whether a series of utterances implies a new fact about the last speaker.
    Builds on the MSC_Turns dataset
    """

    def __init__(self, path, tokenizer=None, len_context=2, speaker_prefixes=None, nofact_token='', max_samples=None, batch_format="huggingface", batch_pad_id=0):
        super().__init__(path, tokenizer, len_context, speaker_prefixes, nofact_token, max_samples, batch_format, batch_pad_id)
        assert self.batch_format == 'huggingface', "batch_format '{}' is incompatible with class {}".format(self.batch_format, self.__class__.__name__)

    def __getitem__(self, index):
        """
        Returns two sentences and boolean indicating whether the sentences imply a new fact
        First sentence is composed of all utterances in the context, except the last (so maximum of 'len_context' - 1 utterances);
        Second sentence is last utterance in context
        """
        turn = self.turns[index]
        if self.speaker_prefixes is not None:
            turn_0 = ' '.join([self.speaker_prefixes[p] + ' ' + t for p, t in turn[:-1]])
            turn_1 = self.speaker_prefixes[turn[-1][0]] + ' ' + turn[-1][1]
        else:
            turn_0 = ' '.join([t for _, t in turn[:-1]])
            turn_1 = turn[-1][1]
        has_fact = self.personas[index] != self.nofact_token
        return turn_0, turn_1, has_fact

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


    def evaluate(self, model, device="cpu", batch_size=1):

        def print_predictions(data, pred):

            for (turn_0, turn_1, has_fact), pred_fact in zip(data, pred):
                print('context:    ', turn_0)
                print('last turn:  ', turn_1)
                print('target:     ', has_fact)
                print('prediction: ', pred_fact.item())
                print('-' * 40)

        model = model.to(device)
        model.eval()
        all_labels = []
        all_preds = []

        for start_index in range(0, len(self.turns), batch_size):
            data = [self.__getitem__(start_index + i) for i in range(batch_size) if start_index + i < len(self.turns)]
            X, y = self.batchify(data)
            X = (x.to(device) for x in X)

            with torch.no_grad():
                logits = model(*X).cpu()
                pred = logits.argmax(dim=1)
                
            all_labels.append(y)
            all_preds.append(pred)
            print_predictions(data, pred)

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        stats = {
            "test_acc": binary_accuracy(all_preds, all_labels).item(),
            "f1": binary_f1_score(all_preds, all_labels).item(),
            "cm": binary_confusion_matrix(all_preds, all_labels).tolist()
        }

        return stats

if __name__ == "__main__":

    from transformers import AutoTokenizer

    # Define setup
    datapath = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/session_1/train.txt'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    speaker_prefixes=["[me]", "[you]"]
    add_tokens = None

    # Prepare for test
    if add_tokens is not None:
        num_added_toks = tokenizer.add_tokens(add_tokens)
    
    # Test extraction of dialogue turns and persona facts
    msc_turns = MSC_Turn_Facts(datapath, tokenizer=tokenizer, len_context=4, speaker_prefixes=speaker_prefixes)

    batch = [msc_turns[i] for i in range(10)]

    for item in batch:
        print(item[0])
        print(item[1])
        print(item[2])
        print('-'*40)

    print(msc_turns.batchify(batch))
    print('-' * 40)
