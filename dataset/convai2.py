###
### Class to read the ConvAI2 / PersonaChat dataset, and preprocess the data.
###

from torch.utils.data import Dataset
import utils.logging as logging


CONVAI2_PERSONAS = ['none', 'self', 'other', 'both']
CONVAI2_VARIANTS = ['original', 'revised']
CONVAI2_CANDS = ['no_cands']

class ConvAI2(Dataset):
    
    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('ConvAI2')
        group.add_argument("--convai2_version", default=['both', 'revised', 'no_cands'], nargs='*', help="Keywords to identify the desired ConvAI2 dataset file")
        return parser

    def __init__(self, basedir='./', version=['both', 'revised', 'no_cands'], subset='train'):
        super(ConvAI2, self).__init__()
        assert len(version) in [2, 3], f"Convai file version should have 2 or 3 components, picked from {CONVAI2_PERSONAS}, {CONVAI2_VARIANTS} and {CONVAI2_CANDS}"
        assert version[0] in CONVAI2_PERSONAS, f"Invalid version component '{version[0]}'; should be one of {CONVAI2_PERSONAS}."
        assert version[1] in CONVAI2_VARIANTS, f"Invalid version component '{version[1]}'; should be one of {CONVAI2_VARIANTS}."
        if len(version) == 3:
            assert version[2] in CONVAI2_CANDS, f"Invalid version component '{version[2]}'; should be one of {CONVAI2_CANDS}."
        self.dialogues = self.parse_dialogs(basedir, version, subset)

    def parse_dialogs(self, basedir, version, subset):

        len_self_prefix, len_other_prefix = len("your persona: "), len("partner's persona: ")
        dialogues = []
        with open(basedir + subset + '_' + '_'.join(version) + '.txt', "r") as f:
            lines = list(f)
            i = 0
            while i < len(lines):

                # Parse lines for one dialogue
                personas_self, personas_other, turns = [], [], []
                line_split = lines[i].split(sep=' ', maxsplit=1)

                # Check if the file contains persona sentences for 'your persona'
                while (line_split[1][:len_self_prefix] == "your persona: ") and (i < len(lines)):
                    personas_self.append(line_split[1][len_self_prefix:-1])  # do not include the last '\n'
                    i += 1
                    if i >= len(lines): break
                    line_split = lines[i].split(sep=' ', maxsplit=1)

                # Check if the file contains persona sentences for 'partner's persona'
                while (line_split[1][:len_other_prefix] == "partner's persona: ") and (i < len(lines)):
                    personas_other.append(line_split[1][len_other_prefix:-1]) # do not include the last '\n'
                    i += 1
                    if i >= len(lines): break
                    line_split = lines[i].split(sep=' ', maxsplit=1)
                
                # Check again if the file contains persona sentences for 'your persona', because sometimes partner's persona comes before your persona
                while (line_split[1][:len_self_prefix] == "your persona: ") and (i < len(lines)):
                    personas_self.append(line_split[1][len_self_prefix:-1])  # do not include the last '\n'
                    i += 1
                    if i >= len(lines): break
                    line_split = lines[i].split(sep=' ', maxsplit=1)
                
                # Collect the dialogue utterances
                prev_dialog_line_nr = 0
                while (int(line_split[0]) > prev_dialog_line_nr) and (i < len(lines)):
                    utterances = line_split[1].split('\t')[:2] # Cut off the candidate sentences (if they are present)
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
                    'personas': [personas_self, personas_other], 
                    'dialog': turns
                })

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
    subset = 'valid'

    convai2_dialogs = ConvAI2(
        basedir=basedir, 
        version=version, 
        subset=subset, 
    )

    data = [convai2_dialogs[i] for i in range(10)]

    for item in data:
        logging.verbose('\n'.join([ str(k) + ': ' + str(v) for k, v in item.items()]))
        logging.verbose('-'*40)
