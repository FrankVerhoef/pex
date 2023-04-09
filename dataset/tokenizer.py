from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from tokenizers.processors import TemplateProcessing

END_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
SPECIAL_TOKENS = [END_TOKEN, UNK_TOKEN, PAD_TOKEN]

def train_tokenizer(corpus, max_size):

    tokenizer = Tokenizer(models.WordPiece(unk_token=UNK_TOKEN))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(), 
        normalizers.Lowercase(), 
        normalizers.StripAccents()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(), 
        pre_tokenizers.Punctuation()
    ])

    trainer = trainers.WordPieceTrainer(vocab_size=max_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single= "$A " + END_TOKEN,
        special_tokens=[
            (END_TOKEN, tokenizer.token_to_id(END_TOKEN)),
        ],
    )

    return tokenizer


if __name__ == '__main__':

    from dataset.msc_summary_turns import MSC_Turns

    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_personasummary/'
    msc_turns = MSC_Turns(basedir=basedir, sessions=[1], subset='train', tokenizer=None, len_context=3, speaker_prefixes=None, max_samples=1000)

    tokenizer = train_tokenizer(msc_turns.corpus(), max_size=4000)
    enc = tokenizer.encode("Hey, how are you doing?")
    print(enc.ids)
    print(tokenizer.encode("").ids)


