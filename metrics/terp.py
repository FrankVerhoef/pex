###
### Functions to calculate the TERp metric for combinations of predicted summaries and target summaries
### Uses external JAVA program to calculate the TERp scores
### See: https://github.com/snover/terp/blob/master/README.md
###

import torch
from torchmetrics import Metric

import json
import itertools
import subprocess, os

import utils.logging as logging
from utils.plotting import plot_heatmap


class TerpMetric(Metric):

    terp_command = "bin/terpa -r {ref_path} -h {hyp_path} -o sum,pra,nist,html"
    trans_format = "{sentence} ([sys][{seg:06d}][{id:06d}])\n"
    ref_file = "ref.trans"
    hyp_file = "hyp.trans"
    terpscores_file = ".seg.scr"
    terp_dir = None
    java_home = None
    tmp_dir = None

    @classmethod
    def set(cls, terp_dir, java_home, tmp_dir):
        cls.terp_dir = terp_dir
        cls.java_home = java_home
        cls.tmp_dir = tmp_dir

    @classmethod
    def add_cmdline_args(cls, parser):
        group = parser.add_argument_group('TerpMetric')
        group.add_argument("--java_home", type=str, required=True, help="Java home directory")
        group.add_argument("--terpdir", type=str, required=True, help="Root directory of TERp program")
        group.add_argument("--tmpdir", type=str, required=True, help="Temp directory to store intermediate results")
        return parser

    def __init__(self):
        super().__init__()
        self.add_state("info", default=[], dist_reduce_fx="cat", persistent=False)
        self.add_state("targets", default=[], dist_reduce_fx="cat", persistent=False)
        self.add_state("predictions", default=[], dist_reduce_fx="cat", persistent=False)

    def update(self, id, preds, targets):
        assert len(preds) == len(targets)
        self.info.append({"id": id, "length": len(targets)})
        self.targets.append(targets)
        self.predictions.append(preds)

    def compute(self):
        
        # Check if metric has been configured correctly
        assert self.terp_dir is not None
        assert self.tmp_dir is not None
        assert self.java_home is not None

        # Define paths to input and output files
        ref_path = self.tmp_dir + self.ref_file
        hyp_path = self.tmp_dir + self.hyp_file
        terpscores_path = self.terp_dir + self.terpscores_file

        # Create a reference file and hypothesis file in TRANS format, as input for the TERp program
        self._to_trans_files(ref_path, hyp_path)

        # Execute TERp calculation in appropriate environment
        # The results are written to the file TERP_SCORES_FILENAME, which is stored in the root directory of the TERp program (TERP_DIR)
        env = os.environ.copy()
        env["JAVA_HOME"] = self.java_home
        completed_process = subprocess.run(
            self.terp_command.format(ref_path=ref_path, hyp_path=hyp_path).split(),
            cwd=self.terp_dir,
            env=env,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            check=True
        )
        logging.spam("TERp output\n" + completed_process.stdout.decode())

        # Collect the statistics from the resultsfile
        stats = self._get_stats(terpscores_path)

        return stats
    
    def _to_trans_files(self, ref_path, hyp_path):
        """
        Create a reference file and hypothesis file in TRANS format, as input for the TERp program
        """
        refs = open(ref_path, 'w')
        hyps = open(hyp_path, 'w')
        for seg, (preds, targets) in enumerate(zip(self.predictions, self.targets)):
            for i, (pred, target) in enumerate(zip(preds, targets)):
                refs.write(self.trans_format.format(sentence=target, seg=seg, id=i))
                hyps.write(self.trans_format.format(sentence=pred, seg=seg, id=i))
        refs.close()
        hyps.close()

    def _get_stats(self, terp_scores_path):
        """
        Reads the results file created by the TERp program and returns the results in a dict with a tensor of TERp scores per id
        """
        results = {}

        # Collect the individual scores in a dict
        with open(terp_scores_path) as results_file:
            for item_info in self.info:
                id = item_info["id"]
                results[id] = []
                for _ in range(item_info["length"]):
                    line = results_file.readline()

                    # Split line in components. Each line in the results file has format: 
                    # <sys> <seg> <id> <terp score> <number of words>
                    terp_score = float(line.split()[3])

                    # Append score to current results_list
                    results[id].append(terp_score)
                
                # Convert to tensor
                results[id] = torch.tensor(results[id])

        return results      

###
### TERp calculation as a series of functions
###


# Info required to run the TERp program
TERP_DIR = '/Users/FrankVerhoef/Programming/terp/'
JAVA_HOME = '/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home'
TERP_COMMAND = "bin/terpa -r {ref_path} -h {hyp_path} -o sum,pra,nist,html"

# Format for the input files required by the TERp program (TRANS-format)
TRANS_FORMATSTRING = "{sentence} ([session_{session}/{subset}.txt][{dialog_id:06d}][{combination_id:06d}])\n"

# Filename used by TERp program to store the results (is determined by TERp program)
TERPSCORES_FILENAME = ".seg.scr"

# Names for intermediate files used for TERp calculation and to store labels for the heatmaps
REFERENCES_FILENAME = "ref.trans"
HYPOTHESES_FILENAME = "hyp.trans"
EVAL_FILENAME = "eval.txt"

# Threshold for considering the score a 'good' match
TERP_MAXIMUM = 0.6


def calc_terp_stats(predicted_summaries, target_summaries, session, subset, indices, savedir='./'):

    # Define paths to input and output files
    ref_path = savedir + REFERENCES_FILENAME
    hyp_path = savedir + HYPOTHESES_FILENAME
    eval_path = savedir + EVAL_FILENAME
    terpscores_path = TERP_DIR + TERPSCORES_FILENAME

    # Create the input files needed for the TERp program
    _prepare_terp_files(predicted_summaries, target_summaries, session, subset, indices, ref_path, hyp_path, eval_path)

    # Execute TERp calculation in appropriate environment
    # The results are written to the file TERP_SCORES_FILENAME, which is stored in the root directory of the TERp program (TERP_DIR)
    env = os.environ.copy()
    env["JAVA_HOME"] = JAVA_HOME
    completed_process = subprocess.run(
        TERP_COMMAND.format(ref_path=ref_path, hyp_path=hyp_path).split(),
        cwd=TERP_DIR,
        env=env,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        check=True
    )
    logging.spam("TERp output\n" + completed_process.stdout.decode())

    # Summarize the statistics and save heatmaps to visualize results
    stats = _get_stats_and_save_heatmaps(eval_path, terpscores_path, session, subset, indices, savedir)

    return stats


def _prepare_terp_files(predicted_summaries, target_summaries, session, subset, indices, ref_path, hyp_path, eval_path):
    """
    Dump predictions and targets in two files in 'TRANS'-format, 
    These output files can be used to calculate TERp scores.
    Also save predictions and targets in seperate file for later use (e.g. for labels in heatmaps).
    """
    refs = open(ref_path, 'w')
    hyps = open(hyp_path, 'w')
    evals = open(eval_path, "w")

    for dialog_id, prediction, target in zip(indices, predicted_summaries, target_summaries): 

        # Split the prediction and target sentences on '.' to facilitate matching
        pred_sentences = [p.lower() for p in prediction.replace('. ', '\n').replace('.', '').split('\n') if p != '']
        target_sentences = [t.lower() for t in target.replace('. ', '\n').replace('.', '').split('\n') if t != '']

        # Make list of all combinations between prediction and target sentences
        combinations = list(itertools.product(pred_sentences, target_sentences))

        # Create a reference file and hypothesis file in TRANS format, as input for the TERp program
        for c, (pred, target) in enumerate(combinations):
            refs.write(TRANS_FORMATSTRING.format(sentence=target, session=session, subset=subset, dialog_id=dialog_id, combination_id=c))
            hyps.write(TRANS_FORMATSTRING.format(sentence=pred, session=session, subset=subset, dialog_id=dialog_id, combination_id=c))

        # Save the prediction and target sentences for later use (e.g. as labels in the heatmaps)
        evals.write(json.dumps({"dialog_id": dialog_id, "prediction": pred_sentences, "target": target_sentences}) + '\n')

    refs.close()
    hyps.close()
    evals.close()


def _read_terp_results(terp_results_file, session, subset, indices):
    """
    Reads the results file created by the TERp program and returns the results in a dict, 
    with the dialog_id as key and a tensor with the scores for one dialog as value.
    NOTE: the results are ordered by TERp program on dialog number, which may be in different order than the supplied indices!
    """
    results_dict = {}
    current_dialog_id = -1

    # Collect the individual scores in a dict
    with open(terp_results_file) as results_file:
        for line in results_file:

            # Split line in components. Each line in the results file has format: 
            # <session/subset> <dialog number> <pred-target combination number> <terp score> <number of words>
            session_and_subset, dialog_id, combination_id, terp_score, _ = line.split()
            dialog_id = int(dialog_id)
            combination_id = int(combination_id)
            terp_score = float(terp_score)

            # Verify session and subset
            assert session_and_subset == f"session_{session}/{subset}.txt", "Expected session/subset {session}/{subset} does not match with line: {line}"

            # If dialog_id is different from previous --> scores belong to new dialog
            if dialog_id != current_dialog_id:
                current_dialog_id = dialog_id
                results_dict[current_dialog_id] = []

            # Check that the combination_id corresponds with length of current results_list
            assert len(results_dict[current_dialog_id]) == combination_id, \
                f"Expected combination_id {len(results_dict[current_dialog_id])} does not match with line: {line}"

            # Append score to current results_list
            results_dict[current_dialog_id].append(terp_score)

    # Check if dialog_id's correspond with the provided indices
    assert len(set(indices).intersection(set(results_dict.keys()))) == len(indices), "Mismatch between dialog_id's in resultsfile and provided indices"

    # Convert lists to tensors
    results_dict = {dialog_id: torch.tensor(terp_scores) for dialog_id, terp_scores in results_dict.items()}

    return results_dict


def _get_stats_and_save_heatmaps(eval_file, terp_results_file, session, subset, indices, savedir='./'):

    # Load predictions and targets from eval_file and corresponding terp scores from terp_results_file
    with open(eval_file, "r") as f:
        eval_list = [json.loads(line) for line in f]
    terp_results = _read_terp_results(terp_results_file, session, subset, indices)

    # Initialise metrics lists
    terp_f1s= []
    terp_precisions = []
    terp_recalls = []

    # Loop over all evaluation samples (dialog_id, prediction, target) and corresponding scores to calculate recall, precision and f1
    for eval in eval_list:

        dialog_id = eval["dialog_id"]
        pred_sentences = eval["prediction"]
        target_sentences = eval["target"]
        terp_scores = terp_results[dialog_id].view(len(eval["prediction"]), len(eval["target"]))

        # Calculate precision, recall and f1, using a threshold for terp_scores of TERP_MAXIMUM
        matching_predictions = terp_scores <= TERP_MAXIMUM
        terp_precision = torch.any(matching_predictions, dim=1).float().mean().item()
        terp_recall = torch.any(matching_predictions, dim=0).float().mean().item()
        terp_f1 = (2 * terp_precision * terp_recall / (terp_precision + terp_recall)) if (terp_precision + terp_recall) != 0 else 0

        # Append metrics to metric lists
        terp_f1s.append(terp_f1)
        terp_precisions.append(terp_precision)
        terp_recalls.append(terp_recall)

        # Save corresponding heatmap
        im = plot_heatmap(
            scores=terp_scores.permute(1,0), 
            threshold=TERP_MAXIMUM,
            criterion = lambda x, threshold: x <= threshold,
            targets=target_sentences, 
            predictions=pred_sentences, 
            title=f"TERp heatmap MSC_Summary session_{session}/{subset}, dialog {dialog_id}\n(threshold={TERP_MAXIMUM:.2f})"
        )
        im.figure.savefig(f"{savedir}terp_heatmap_session_{session}_{subset}_{dialog_id:06d}.jpg")

    # Collect and return stats
    stats = {
        "terp_f1": terp_f1s,
        "terp_precision": terp_precisions,
        "terp_recall": terp_recalls,
    }
    return stats 


def terp_summary(scores, threshold, num_pred, num_tgt):
    """
    Calculate F1, precision and recall for a series of prediction sentences and target sentences
    """
        
    scores = scores.view(num_pred, num_tgt).permute(1,0)
    matching_predictions = scores <= threshold
    precision = torch.any(matching_predictions, dim=0).float().mean().item()
    recall = torch.any(matching_predictions, dim=1).float().mean().item()
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {"scores": scores.tolist(), "f1": f1, "precision": precision, "recall": recall}