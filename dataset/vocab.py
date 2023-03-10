import spacy
from spacy.symbols import ORTH
from collections import Counter
from tqdm import tqdm
import json

nlp = spacy.load("en_core_web_sm", disable = ['ner', 'tagger', 'parser', 'textcat'])

START_TOKEN = '<SOS>'
END_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
special_tokens = [START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN]


class Vocab:

    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")
        return parser

    def __init__(self):

        # initialize empty vocab with special tokens
        self.tok2ind = dict()
        self.ind2tok = []
        self.count = Counter()
        self.special_tokens = set()
        self.add_special_tokens(special_tokens)

    def __len__(self):
        return len(self.ind2tok)

    def __call__(self, s):
        return self.tokenize(s)

    def tokenize(self, s):
        tokens = [
            (token.text.lower() if not token.text in self.special_tokens else token.text) 
            for token in nlp.tokenizer(s)
        ]
        return tokens

    def encode(self, s, append_end_token=False):
        tokens = self.tokenize(s)
        if append_end_token:
            tokens.append(self.tok2ind[END_TOKEN])
        vec = self.convert_tokens_to_ids(tokens)
        return vec

    def decode(self, vec, **kwargs):
        skip_special_tokens = kwargs.get('skip_special_tokens', False)
        tokens = self.convert_ids_to_tokens(vec)
        text = ' '.join([token for token in tokens if (skip_special_tokens == False) or (token not in self.special_tokens)])
        return text

    def convert_tokens_to_ids(self, tokens):
        return [self.tok2ind.get(t, self.tok2ind['<UNK>']) for t in tokens]

    def convert_ids_to_tokens(self, vec):
        return [self.ind2tok[i] for i in vec]

    def add_special_tokens(self, extra_tokens):
            
        for t in extra_tokens:
            self.tok2ind[t] = len(self.ind2tok)
            self.ind2tok.append(t)
            nlp.tokenizer.add_special_case(t, [{ORTH: t}])
        self.special_tokens.update(extra_tokens)

    def add_to_vocab(self, sentences):
        """
        Loop through all sentences, apply tokenizer and add tokens (with count) to vocab
        """
        vocab = Counter()
        # add new tokens
        for s in tqdm(sentences):
            tokens = self.tokenize(s)
            vocab.update(tokens)

        # exclude special tokens
        for t in self.special_tokens:
            if t in vocab:
                del vocab[t]

        self.count.update(vocab)
        old_len = len(self.ind2tok)
        new_tokens = [t for t in vocab.keys() if t not in self.tok2ind.keys()]
        self.ind2tok.extend(new_tokens)
        self.tok2ind.update({t: i + old_len for i, t in enumerate(new_tokens)})

        print("Added {} tokens to vocabulary".format(len(new_tokens)))

    def cut_vocab(self, max_tokens):
        token_count = sum(self.count.values())
        if max_tokens < len(self.ind2tok):
            vocab = [t for t, _ in sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_tokens]]
            self.count = Counter({t: c for t, c in self.count.items() if t in vocab})
            self.ind2tok = list(self.special_tokens)
            self.ind2tok.extend(vocab)
            self.tok2ind = {self.ind2tok[i]: i for i in range(len(self.ind2tok))}
            print("Reduced vocab to {} tokens, covering {:.1%} of corpus".format(max_tokens, sum(self.count.values()) / token_count))

    def save_vocab(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.ind2tok) + '\n')
            f.write(json.dumps(list(self.special_tokens)) + '\n')

    def load_vocab(self, path):
        self.__init__()
        with open(path, "r") as f:
            try:
                self.ind2tok = json.loads(f.readline())
                self.special_tokens = set(json.loads(f.readline()))
                self.tok2ind = {token: index for index, token in enumerate(self.ind2tok)}
                for t in self.special_tokens:
                    nlp.tokenizer.add_special_case(t, [{ORTH: t}])
            except:
                print("Failed to load vocab from ", path)


###
### Test
###

if __name__ == "__main__":
    sentences = [
        "I need some advice on where to go on vacation, have you been anywhere lately?", 
        "<P0>I have been all over the world. I'm military.",  
        "I served or serve in the military. <P1> I've traveled the world."
    ]
    voc = Vocab()
    voc.add_special_tokens(['<P0>', '<P1>'])
    voc.add_to_vocab(sentences)
    print(voc(sentences[0]))

    voc.save_vocab('testvocab')
    voc.load_vocab('testvocab')

    print(voc.tok2ind)