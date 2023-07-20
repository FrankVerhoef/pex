from dataset.msc_sessions import MSC_Session
from models.speechact_clf import SpeechactClassifier
import json

basedir = "data/msc/msc_dialogue/"
checkpoint_dir = "checkpoints/"

subsets = {
    1: ['train', 'valid', 'test'],
    2: ['train', 'valid', 'test'],
    3: ['train', 'valid', 'test'],
    4: ['train', 'valid', 'test'],
    5: ['valid', 'test']
}

config = {
    "speaker_prefixes": ["<other>", "<self>"],
    "sessionbreak_token": "<sessionbreak>",
    "speechact_classifier": SpeechactClassifier(checkpoint_dir=checkpoint_dir, modelname="trained_speechact_bert")
}

variant = {"include_persona": False, "include_history": False}
variant_key = "no_persona_no_hist"

MSC_Session.set(**config)

max_samples = 3

msc_sessions = {}
for session in subsets.keys():
    if session == 1:
        version = ['both', 'revised']
        session = '-'.join(['1'] + version)
    msc_sessions[int(str(session)[0])] = {
        variant_key: {
            subset: MSC_Session(basedir=basedir, session=session, subset=subset, max_samples=max_samples, **variant)
            for subset in subsets[int(str(session)[0])]
        }
    }

m = {
    session: {
        variant_key: {subset: msc_sessions[session][variant_key][subset].measurements() for subset in subsets[session]}
    }
    for session in subsets.keys()
}

with open(basedir + 'session_measurements.json', 'w') as f:
    f.write(json.dumps(m))
