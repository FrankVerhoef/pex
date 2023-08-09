from dataset.msc_sessions import MSC_Session
from models.speechact_clf import SpeechactClassifier
import json
import utils.logging as logging

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

max_samples = None

# Prepare logging
logging.set_log_level("VERBOSE")
logging.add_file_handler(logdir='logs/')

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

logging.info(f"Start extracting speechacts from {basedir}, subsets {list(subsets.keys())}")
m = {}
for session in subsets.keys():
    m[session] = {variant_key: {}}
    for subset in subsets[session]:
        m[session][variant_key][subset] = msc_sessions[session][variant_key][subset].measurements()
        logging.verbose(f"Extracted speechacts from {session}-{subset}")

logging.info("Finished extracting speechacts")
resultsfile = basedir + 'session_measurements.json'
logging.info(f"Saved results in {resultsfile}")
with open(resultsfile, 'w') as f:
    f.write(json.dumps(m))
