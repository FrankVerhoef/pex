###
### Functions based on analysis of the 'rhythm' in a dialogue, based on the sequence and frequency of speechacts.
###

from collections import Counter

def count_speechacts(rhythm):
    return Counter(rhythm.replace('-', ''))

def count_speechpatterns(rhythm):
    return Counter(rhythm.split('-'))

def joint_pattern(pattern, dialogue_rhythm):
    """
    Calculates relative joint frequency of 'pattern' in the dialogue (=count / number of utterances minus 1)
    pattern: string with two sequences of dialogue acts, separated by '-'; example: 'SQ-AE'
    dialogue_rhythm: string with multiple sequences of dialogue acts, seperated by '-', representing the rhythm of a complete dialogue
    """
    acts_1, acts_2 = pattern.split('-')
    utterance_acts = dialogue_rhythm.split('-')
    if len(utterance_acts) > 1:
        pairs = [(set(u1), set(u2)) for u1, u2 in zip(utterance_acts[:-1], utterance_acts[1:])]
        match = lambda u1, u2: len(set(acts_1).intersection(u1)) > 0 and len(set(acts_2).intersection(u2)) > 0
        rel_freq = sum([match(u1, u2) for u1, u2 in pairs]) / len(pairs)
    else:
        rel_freq = 0
    return rel_freq

def conditional_pattern(pattern, dialogue_rhythm):
    """
    Calculates relative conditional frequency of 'pattern' in the dialogue
    pattern: string with two sequences of dialogue acts, separated by '-'; example: 'SQ-AE'
    dialogue_rhythm: string with multiple sequences of dialogue acts, seperated by '-', representing the rhythm of a complete dialogue
    """
    acts_1, acts_2 = pattern.split('-')
    utterance_acts = dialogue_rhythm.split('-')
    rel_freq = -1
    if len(utterance_acts) > 1:
        pairs = [(set(u1), set(u2)) for u1, u2 in zip(utterance_acts[:-1], utterance_acts[1:])]
        count_acts1, count_acts2 = 0, 0
        for u1, u2 in pairs:
            if len(set(acts_1).intersection(u1)) > 0:
                count_acts1 += 1
                if len(set(acts_2).intersection(u2)) > 0:
                    count_acts2 += 1
        rel_freq = count_acts2 / max(count_acts1, 1)
    return rel_freq

