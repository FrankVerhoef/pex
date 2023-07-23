import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def overlap(personas, utterances, pos_tags):
    """
    Takes a list of persona sentences, a list of utterances and a list of POS tags to determine overlapping words.
    Returns: list with for each utterance a boolean indicating whether words of desired postag occur in the persona sentences.
    """
    single_utterance = isinstance(utterances, str)
    if single_utterance:
        utterances = [utterances]
    base_tokens = set([token.lemma_ for token in nlp(' '.join(personas)) if token.pos_ in pos_tags])
    utterances_tokens = [
       set([token.lemma_ for token in nlp(utterance) if token.pos_ in pos_tags])
       for utterance in utterances
    ]
    overlap = [len(base_tokens.intersection(u)) > 0 for u in utterances_tokens]
    return overlap[0] if single_utterance else overlap