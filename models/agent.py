##
## Simple class for a dialogue agent
##

from models.memory import TextMemory
import utils.logging as logging

class Agent:
    """
    Simple class for a dialogue agent with memory.
    """

    MEMORY_SIZE = 1000
    SELF = '<self>'
    OTHER = '<other>'

    @classmethod
    def set(cls, memory_size=1000, prefix_other=None, prefix_self=None):
        cls.MEMORY_SIZE = memory_size
        cls.OTHER = prefix_other
        cls.SELF = prefix_self

    def __init__(self, id, generator, persona=None):
        """
        Initialize agent and add persona sentences to memory
        """
        self.id = id
        self.generator = generator
        self.mem = {id: TextMemory(id, maxsize=self.MEMORY_SIZE)}
        self.add_persona(self.id, persona)
        self.dialogues = {}

    def add_persona(self, speaker_id, persona):

        if speaker_id not in self.mem.keys():
            self.mem[speaker_id] = TextMemory(id, maxsize=self.MEMORY_SIZE)
        if persona is not None:
            for p in persona:
                self.mem[speaker_id].add(p)

    def observe(self, speaker_id, message):
        
        if speaker_id in self.dialogues.keys():
            self.dialogues[speaker_id].append((speaker_id, message))
        else:
            self.dialogues[speaker_id] = [(speaker_id, message)]

    def act(self, speaker_id, forced_text=None):

        if speaker_id not in self.dialogues.keys():
            self.dialogues[speaker_id] = []

        if forced_text is not None:
            response = forced_text
        else:
            # Construct context for the generator
            context = '\n'.join([self.SELF + m for m in self.mem[self.id].recall()]) + '\n'
            if speaker_id in self.mem.keys():
                context += '\n'.join([self.OTHER + m for m in self.mem[speaker_id].recall()]) + '\n'
            context += '\n'.join([(self.SELF if s_id == self.id else self.OTHER) + text for s_id, text in self.dialogues[speaker_id]]) + '\n'

            # Get response from generator
            response = self.generator(context)
            response = response.split(self.OTHER)[0].replace(self.SELF, '')

        # Add to dialogue memory and return response
        self.dialogues[speaker_id].append((self.id, response))
    
        return response

    def __repr__(self):
        return f"<Agent id: {self.id}, {len(self.mem.keys())} memory keys, {len(self.dialogues.keys())} dialogue keys>"

    def __str__(self):
        s = f"Agent {self.id}\n"
        s += f"Memory: {len(self.mem.keys())}\n"
        s += '\n'.join([
            str(memory) 
            for memory in self.mem.values()
        ]) + '\n'
        s += f"Dialogues: {len(self.dialogues.keys())}\n"
        s += '\n'.join([
            f"{other_id}:\n" + '\n'.join([(self.SELF if s_id == self.id else self.OTHER) + text for s_id, text in dialogue])
            for other_id, dialogue in self.dialogues.items()
        ]) + '\n'
        return(s)


if __name__ == '__main__':

    import torch
    import argparse
    from models.dialogpt import DialoGPT
    from dataset.msc_sessions import MSC_Session
    from transformers import AutoTokenizer, GenerationConfig
    from functools import partial
    from utils.general import load_config
    import random

    parser = argparse.ArgumentParser(description="Test Agent")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser = DialoGPT.add_cmdline_args(parser)
    parser = MSC_Session.add_cmdline_args(parser)

    args = parser.parse_args()
    print(vars(args))

    ##
    ## Settings for this test
    ##

    # Model
    args.load = 'test_dgpt'
    args.checkpoint_dir = '/Users/FrankVerhoef/Programming/PEX/checkpoints/'

    # Dataset
    args.datadir = '/Users/FrankVerhoef/Programming/PEX/data/'
    args.basedir = 'msc/msc_dialogue/'
    args.session = 2
    args.speaker_prefixes = ['<other>', '<self>']
    args.sessionbreak_token = '<session>'
    args.include_history = False
    args.include_persona = True
    args.augmented = True
    args.selected_turns = [3]
    args.persona_selector = None
    args.input_order = 'history-personas-current'
    args.test_samples = 10
    dialog_id = 1
    num_turns = 8

    # Generation setting
    args.do_sample = True
    args.num_beams = 5
    args.temperature = 1.5
    args.top_p = 0.9
    args.top_k = 50
    args.decoder_max = 30

    ##
    ## Initialize model
    ##
    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.bos_token_id = tokenizer.eos_token_id
    if args.speaker_prefixes is not None:
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(args.speaker_prefixes[0])
    model = DialoGPT(args.lm, tokenizer.bos_token_id)
    model.model.resize_token_embeddings(len(tokenizer))

    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        logging.info("Loading model from {}".format(loadpath))
        model.load_state_dict(torch.load(loadpath, map_location=torch.device(args.device)))

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.decoder_max,
    )
    generator = partial(MSC_Session.predict, model=model, generation_config=generation_config, device=args.device, batch_size=1)

    ##
    ## Initialize agents with persona information and memory
    ## 

    # Configure the dataset
    MSC_Session.set(tokenizer=tokenizer, speaker_prefixes=args.speaker_prefixes, sessionbreak_token=args.sessionbreak_token)
    dataset_config = {
        'basedir': args.datadir + args.basedir,
        'session': args.session if args.session != 1 else '-'.join(['1'] + args.convai2_version),
        'include_persona': args.include_persona,
        'include_history': args.include_history,
        'augmented': args.augmented, 
        'selected_turns': args.selected_turns,
        'persona_selector': args.persona_selector,
        'input_order': args.input_order
    }
    testdata = MSC_Session(subset='test', max_samples=args.test_samples, **dataset_config)

    # Get the persona information
    persona_1 = testdata.personas(dialog_id, "Speaker 1")
    persona_2 = testdata.personas(dialog_id, "Speaker 2")

    # Initialize the agents
    agents = [
        Agent(id="Mike", generator=generator, persona=persona_1),
        Agent(id="John", generator=generator, persona=persona_2)
    ]

    ##
    ## Start the dialogue
    ##

    # Current dialogue consists of all utterances after the last sessionbreak (speaker == 'Nobody')
    speakers = [s for s, _ in testdata.history[dialog_id]]
    start_index = -(speakers[::-1].index('Nobody'))   # This is the last occurrance of a sessionbreak
    current_dialogue = testdata.history[dialog_id][start_index:] if start_index < 0 else []

    # Start with 'forcing' the current dialogue history
    if len(current_dialogue) > 0:
        print("Dialogue history (forced)")
        for speaker_id, utterance in current_dialogue:
            a = int(speaker_id == 'Speaker 2')
            response = agents[a].act(agents[1 - a].id, forced_text=utterance)
            print(f"{agents[a].id.ljust(8)}: {response}")
            agents[1 - a].observe(agents[a].id, response)
        a = 1 - a   # switch turn to other speaker after the forced dialogue start
    else:
        a = random.randint(0,1)

    # Continue conversation for a number of turns
    print("Generated dialogue continuation")
    for _ in range(num_turns):
        response = agents[a].act(agents[1 - a].id)
        print(f"{agents[a].id.ljust(8)}: {response}")
        agents[1 - a].observe(agents[a].id, response)
        a = 1 - a

    for agent in agents:
        print(agent)