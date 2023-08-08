# PEX = Persona EXtractor

This README is still under construction

## Organization

The code is organized in the following folders:

| Folder      | Description |
| ----------- | --- |
| run         | Contains main.py and tune.py the main scripts for either training/evaluation models (with main), or for hyperparameter search |
| dataset     | Classes for loading and preprocessing datasets, such as the variants of the Multi-Session Chat dataset and the SpeechAct dataset |
| models      | Classes that define the models used for persona extraction and dialogue generation |
| metrics     | Classes that define additional (tailor made) metrics e.g. the TERp metric and the NLI metric |
| utils       | Contains scripts with various utility functions such as for loading, plotting, printing, saving |
| notebooks   | Contains several Jupyter notebooks for short tests and for inspection and visualization of results |
| test        | Short Python scripts to check/verify specific functionality

Additional folders are defined for several types of input or output:

| Folder        | Description |
| ------------- | --- |
| data          | Contains the original (downloaded) datasets |
| checkpoints   | Folder used to save or load models |
| logs          | Folder used for storing logs files |
| output        | Folder to save generated output, such as the statistics and evaluation results |

Lastly there are folders for miscelaneous other objectives.

| Folder        | Description |
| ------------- | --- |
| slurm         | Folder with jobscript to run the code on Snellius |
| docs          | Folder with documents, images and other material used for writing my thesis |

## Running the code

### Requirements

### Explanation of the command line arguments

## The notebooks
The notebooks folders contains several notebooks that I have used for short tests and for inspection and visualization of results

| Notebook                  | Description |
| ------------------------- | --- |
| analyse_bart_persona_eval | Inspection and visualization of evaluation results of persona extraction |
| analyse_bart_summary_eval | Inspection and visualization of evaluation results of summary generation |


