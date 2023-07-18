import torch
import numpy as np
from functools import partial

from transformers import AutoTokenizer
from models.bert_classifier import PrefixBert
from dataset.msc_speechact import MSC_SpeechAct

class SpeechactClassifier:

    def __init__(self, checkpoint_dir, modelname, device='cpu'):

        MSC_SpeechAct.set(tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'))
        model = PrefixBert('bert-base-uncased', prefix_size=0, num_classes=len(MSC_SpeechAct.classes.keys()))
        model.load_state_dict(torch.load(checkpoint_dir + modelname, map_location=torch.device(device)))
        self.classifier = partial(MSC_SpeechAct.predict, model=model)

    def __call__(self, sentence_or_sentencelist):
        return self.classifier(sentence_or_sentencelist)

    def get_speechact(self, sentence):
        act = self(sentence)
        return act

    def get_speechacts(self, utterance):
        acts = ''.join(self(MSC_SpeechAct.split_utterance(utterance)))
        return acts

    def get_dialogue_rhythm(self, dialogue):

        # Split utterances in sentences and flatten the list
        split_dialogue = [MSC_SpeechAct.split_utterance(utterance) for utterance in dialogue]
        sentencelist = [sentence for utterance_list in split_dialogue for sentence in utterance_list]

        # Get the speechacts for all sentences (batched for efficiency)
        all_acts = self(sentencelist)
        
        # Regroup acts per utterance
        offset = np.cumsum([0] + [len(utterance_list) for utterance_list in split_dialogue])
        rhythm = '-'.join([''.join(all_acts[start:end]) for start, end in zip(offset[:-1], offset[1:])])

        return rhythm


if __name__ == '__main__':
    from dataset.msc_sessions import MSC_Session
    from utils.speechacts import joint_pattern, conditional_pattern

    # Test extraction of dialogue turns and persona sentences
    msc_sessions = MSC_Session(
        basedir='/Users/FrankVerhoef/Programming/PEX/data/msc/msc_dialogue/', 
        session=3, 
        subset='train',
        include_persona=False,
        include_history=True,
        max_samples=10
    )

    speechact_classifier = SpeechactClassifier(checkpoint_dir='/Users/FrankVerhoef/Programming/PEX/checkpoints/', modelname='trained_speechact_bert')

    for i in range(len(msc_sessions)):
        dialogue, _ = msc_sessions[i]
        rhythm = speechact_classifier.get_dialogue_rhythm(dialogue[:-1].split('\n'))
        print('\n'.join([
            f"{pattern} :\t{utterance}" 
            for pattern, utterance in zip(
                rhythm.split('-'),
                dialogue[:-1].split('\n')
            )]))
        print(rhythm)
        print('G-G : ', joint_pattern('G-G', rhythm))
        print('Q-A : ', conditional_pattern('Q-A', rhythm))
        print('-' * 40)
