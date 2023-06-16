###
### Class to read the ConvAI2 / PersonaChat dataset, and preprocess the data.
###

from torch.utils.data import Dataset
import utils.logging as logging


CONVAI2_PERSONAS = ['none', 'self', 'other', 'both']
CONVAI2_VARIANTS = ['original', 'revised']
CONVAI2_CANDS = ['no_cands']

CONVAI2_TRAIN_VALID_SPLIT = 0.9
CONVAI2_INFO = {
    'train': "For ConvAI2 dataset, use {:.0%} of train dataset for training (rest is available as validation dataset)".format(CONVAI2_TRAIN_VALID_SPLIT),
    'valid': "For ConvAI2 dataset, use {:.0%} of train dataset as validation dataset".format(1 - CONVAI2_TRAIN_VALID_SPLIT),
    'test': "For ConvAI2 dataset, use validation dataset as test dataset"
}

class ConvAI2(Dataset):
    
    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('ConvAI2')
        group.add_argument("--convai2_version", default=['both', 'revised', 'no_cands'], nargs='*', help="Keywords to identify the desired ConvAI2 dataset file")
        return parser

    def __init__(self, basedir='./', version=['both', 'revised', 'no_cands'], subset='train'):
        super(ConvAI2, self).__init__()
        logging.info(f"Init ConvAI2 with basedir={basedir}, version={version}, subset={subset}")
        assert len(version) in [2, 3], f"Convai file version should have 2 or 3 components, picked from {CONVAI2_PERSONAS}, {CONVAI2_VARIANTS} and {CONVAI2_CANDS}"
        assert version[0] in CONVAI2_PERSONAS, f"Invalid version component '{version[0]}'; should be one of {CONVAI2_PERSONAS}."
        assert version[1] in CONVAI2_VARIANTS, f"Invalid version component '{version[1]}'; should be one of {CONVAI2_VARIANTS}."
        if len(version) == 3:
            assert version[2] in CONVAI2_CANDS, f"Invalid version component '{version[2]}'; should be one of {CONVAI2_CANDS}."
        self.dialogues = self._get_dialogue_subset(basedir, version, subset)


    def _get_dialogue_subset(self, basedir, version, subset):

        # Adjust subset indicator: use part of train set for validation, and use validation set for testing
        logging.info(CONVAI2_INFO[subset])
        load_subset = {
            'train': 'train',
            'valid': 'train',
            'test': 'valid'
        }[subset]

        # Read and parse the dialog source file
        convai2_dialogues = self._parse_dialogs(basedir, version, load_subset)

        # Determine the range of indices to use
        num_dialogues = len(convai2_dialogues)
        if subset in ['train', 'valid']:
            num_train = int(CONVAI2_TRAIN_VALID_SPLIT * len(convai2_dialogues))
            if subset == 'train':
                indices = range(0, num_train)
            else:
                indices = range(num_train, num_dialogues)
        else:
            indices = range(num_dialogues)

        # Return the requested dialogues
        return [convai2_dialogues[i] for i in indices]


    def _parse_dialogs(self, basedir, version, subset):
        """
        Note about format of the input file!
        It contains sentences with 'your persona', then sentences with 'partner's persona', then dialogue turns
        Each turn has two utterances. THE FIRST UTTERANCE IS FROM PARTNER (=> Speaker 1 corresponds to Partners's persona)!
        """
        len_self_prefix, len_other_prefix = len("your persona: "), len("partner's persona: ")
        dialogues = []
        filepath = basedir + subset + '_' + '_'.join(version) + '.txt'
        no_cands = len(version) == 3
        try:
            with open(filepath, "r") as f:
                lines = list(f)
                i = 0
                while i < len(lines):

                    # Parse lines for one dialogue
                    personas_1, personas_2, turns = [], [], []
                    line_split = lines[i].split(sep=' ', maxsplit=1)

                    # Check if the file contains persona sentences for 'your persona'
                    while (line_split[1][:len_self_prefix] == "your persona: ") and (i < len(lines)):
                        personas_2.append(line_split[1][len_self_prefix:-1])  # do not include the last '\n'
                        i += 1
                        if i >= len(lines): break
                        line_split = lines[i].split(sep=' ', maxsplit=1)

                    # Check if the file contains persona sentences for 'partner's persona'
                    while (line_split[1][:len_other_prefix] == "partner's persona: ") and (i < len(lines)):
                        personas_1.append(line_split[1][len_other_prefix:-1]) # do not include the last '\n'
                        i += 1
                        if i >= len(lines): break
                        line_split = lines[i].split(sep=' ', maxsplit=1)
                    
                    # Check again if the file contains persona sentences for 'your persona', because sometimes partner's persona comes before your persona
                    while (line_split[1][:len_self_prefix] == "your persona: ") and (i < len(lines)):
                        personas_2.append(line_split[1][len_self_prefix:-1])  # do not include the last '\n'
                        i += 1
                        if i >= len(lines): break
                        line_split = lines[i].split(sep=' ', maxsplit=1)
                    
                    # Collect the dialogue utterances
                    prev_dialog_line_nr = 0
                    while (int(line_split[0]) > prev_dialog_line_nr) and (i < len(lines)):
                        utterances = line_split[1].split('\t')[:2]  # Cut off the candidate sentences (if they are present)
                        if no_cands:
                            utterances[1] = utterances[1][:-1]      # Cut off line '\n' if input format is 'no_cands'
                        turns.append({
                            'text': utterances[0],
                            'id': 'Speaker 1'
                        })
                        turns.append({
                            'text': utterances[1],
                            'id': 'Speaker 2'
                        })
                        prev_dialog_line_nr = int(line_split[0])
                        i += 1
                        if i >= len(lines): break
                        line_split = lines[i].split(sep=' ', maxsplit=1)                

                    dialogues.append({
                        'init_personas': [personas_1, personas_2], 
                        'dialog': turns
                    })

        except FileNotFoundError:
            logging.warning(f"ConvAI2 file '{filepath}' not found -> skipped")

        return dialogues
        
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, i):
        return self.dialogues[i]



if __name__ == "__main__":

    logging.set_log_level("SPAM")
    logging.info("Unit test {}".format(__file__))

    basedir = '/Users/FrankVerhoef/Programming/PEX/data/msc/msc_dialogue/ConvAI2/'
    version = ['both', 'original']
    subset = 'test'

    convai2_dialogs = ConvAI2(
        basedir=basedir, 
        version=version, 
        subset=subset, 
    )

    data = [convai2_dialogs[i] for i in range(10)]

    for item in data:
        logging.verbose('\n'.join([ str(k) + ': ' + str(v) for k, v in item.items()]))
        logging.verbose('-'*40)
