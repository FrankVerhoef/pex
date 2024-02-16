# PEX = Persona EXtractor

This repository contains the code I wrote for my thesis for the MSc Artificial Intelligence at the University of Amsterdam: 
[Improving Dialogue Generation in Longer Conversations by Explicitly Modeling Mentalizing and Joint Co-construction](https://www.researchgate.net/publication/373711296_Improving_Dialogue_Generation_in_Longer_Conversations_by_Explicitly_Modeling_Mentalizing_and_Joint_Co-construction).

In my thesis I investigate how dialogue generation in longer dialogues can be improved. Cooperative communication requires that a dialogue agent implements a mentalization approach that distinguishes between a ‘me-belief’, ‘you-belief’ and ‘we-belief’. To this end, I experiment with extracting persona summaries from the dialogue history and using those summaries for the dialogue generation process combined with a shortened version of the dialogue history that focuses on the current dialogue session only. 

While the experimental results cannot be tied directly to better abilities for cooperative communication, this research provides several useful contributions: 
- it shows that using summaries instead of the full dialogue history for dialogue generation is effective; 
- it provides insights and suggestions about the impact of dataset preprocessing on the training process and the impact of choosing the right generation strategy on the produced utterances; 
- it proposes a new evaluation metric based on analysis of the variability of the speech acts in the generated dialogues compared to the variability of speech acts in human dialogue. 

The benefits of the proposed approach are twofold: i) more transparency because the summaries make visible what personal information is used by the conversational agent, and allow correction or deletion by the user; ii) higher efficiency because storing and processing the summaries requires less computational and energy resources than storing and processing the full dialogue history at the start of each utterance.

Please consider citing my work, if you found the provided resources useful.

Below is an explanation about the contents of the repository.
This README is still under construction.

## Organization

The code is organized in the following folders:

| Folder      | Description |
| :---------- | :--- |
| run         | Contains main.py and tune.py the main scripts for either training/evaluation models (with main), or for hyperparameter search |
| dataset     | Classes for loading and preprocessing datasets, such as the variants of the Multi-Session Chat dataset and the SpeechAct dataset |
| models      | Classes that define the models used for persona extraction and dialogue generation |
| metrics     | Classes that define additional (tailor made) metrics e.g. the TERp metric and the NLI metric |
| utils       | Contains scripts with various utility functions such as for loading, plotting, printing, saving |
| notebooks   | Contains several Jupyter notebooks for short tests and for inspection and visualization of results |
| tests       | Short Python scripts to check/verify specific functionality

Additional folders are defined for several types of input or output:

| Folder        | Description |
| :------------ | :--- |
| data          | Contains the original (downloaded) datasets |
| checkpoints   | Folder used to save or load models |
| logs          | Folder used for storing logs files |
| output        | Folder to save generated output, such as the statistics and evaluation results |

Lastly there are folders for miscelaneous other objectives.

| Folder        | Description |
| :------------ | :--- |
| slurm         | Folder with jobscript to run the code on Snellius |
| docs          | Folder with documents, images and other material used for writing my thesis |

## Running the code

### Requirements

### Explanation of the command line arguments

## The notebooks
The notebooks folders contains several notebooks that I have used for short tests and for inspection and visualization of results

| Notebook                  | Description |
| :------------------------ | :-- |
| analyse_bart_persona_eval | Inspection and visualization of evaluation results of persona extraction |
| analyse_bart_summary_eval | Inspection and visualization of evaluation results of summary generation |
| analyse_gpt2_generation_eval | Inspection and visualization of evaluation results of utterence generation |
| analyse_gpt2_selfchat_eval   | Inspection and visualization of evaluation results of selfchats |
| filter_dataset_concepts   | Filter all nouns, pronous and verbs from the Multi-Session Chat dataset. The list is used by the Knowledge Grounded Dialogue Generation (not part of thesis) |
| loaddeberta               | Download the DEBERTA model, this is used for the BERT-score metric |
| test_*                    | Miscelaneous test notebooks to discover or verify functionality of libraries |
| visualize_sessions        | Visualize samples and statistics for dialogues in the Multi-Session Chat dataset |
| visualize_speechacts      | Visualize samples and statistics for the SpeechAct dataset |
| visualize_summaries       | Visualize samples and statistics for summaries in the Multi-Session Chat dataset |
| visualize_turns           | Visualize samples and statistics for the MSC-Segments dataset |



