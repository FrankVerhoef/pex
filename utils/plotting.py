import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch

def heatmap(data, row_labels, col_labels, title="", ax=None, cbar_ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None # ax.figure.colorbar(im, ax=cbar_ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Show title
    ax.set_title(title)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), criterion=None, threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray, torch.Tensor)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if criterion is None:
        criterion = lambda x, threshold: x >= threshold
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    display_bool = False
    if isinstance(valfmt, str):
        if valfmt == 'bool_char':
            display_bool = True
            valfmt = matplotlib.ticker.StrMethodFormatter("{x}")
        else:
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # kw.update(color=textcolors[int(criterion(im.norm(data[i, j]), threshold))])
            if display_bool:
                kw.update(weight=("bold" if data[i, j] else "normal"))
                text = im.axes.text(j, i, valfmt('T' if data[i, j] else 'F', None), **kw)
            else:
                kw.update(weight=["normal", "bold"][int(criterion(im.norm(data[i, j]), threshold))])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def best(values, better_than, dim):
    if better_than(0, 1):
        return torch.min(values, dim=dim, keepdim=True)[0]
    else:
        return torch.max(values, dim=dim, keepdim=True)[0]
    
def match(values, better_than, threshold, dim):
    if better_than(0, 1):
        return best(values, better_than, dim=dim) <= threshold
    else:
        return best(values, better_than, dim=dim) >= threshold

def plot_heatmap(scores, threshold, criterion, targets, predictions, title):

    fig, ((ax_heatmap, ax_recall), (ax_precision, ax_f1)) = plt.subplots(
        nrows=2, ncols=2, figsize=(9,6),
        sharex='col', sharey='row',
        gridspec_kw={'height_ratios': [len(targets), 1], 'width_ratios': [len(predictions), 1]}
    )

    # Cap length of labels (if necessary)
    # targets = [t[:20] for t in targets]
    # predictions = [p[:20] for p in predictions]

    # Define the colormap
    if criterion(0,1):  
        # Reversed map (smaller is better --> low scores get green color, high scores get red color)
        colors_below = plt.cm.RdYlGn(np.linspace(0.85, 0.55, 256))
        colors_above = plt.cm.RdYlGn(np.linspace(0.4, 0.15, 256))
        all_colors = np.vstack((colors_below, colors_above))
    else: 
        # Regular map (bigger is better --> low scores get red color, high scores get green color)
        colors_below = plt.cm.RdYlGn(np.linspace(0.15, 0.4, 256))
        colors_above = plt.cm.RdYlGn(np.linspace(0.55, 0.85, 256))
        all_colors = np.vstack((colors_below, colors_above))
    score_map = colors.LinearSegmentedColormap.from_list('score_map', all_colors)
    divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=1)

    # Make and annotate the heatmap
    scores = torch.tensor(scores)
    im_heatmap, cbar = heatmap(scores, targets, predictions, title=title, ax=ax_heatmap, cbar_ax=ax_recall, cmap=score_map, norm=divnorm, cbarlabel="Score")
    texts = annotate_heatmap(im_heatmap, valfmt="{x:.1f}", threshold=threshold, criterion=criterion)

    # Calculate precision and recall for the summary bars
    precision = torch.any(criterion(scores, threshold), dim=0).view(1,-1).float()
    recall = torch.any(criterion(scores, threshold), dim=1).view(-1,1).float()
    avg_precision = precision.mean().item()
    avg_recall = recall.mean().item()
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0

    # Display the summary bars with precision and recall
    im_precision = ax_precision.imshow(best(scores, criterion, dim=0), cmap=score_map, norm=divnorm)
    texts = annotate_heatmap(im_precision, data=match(scores, criterion, threshold, dim=0), valfmt="bool_char", threshold=threshold, criterion=criterion)
    im_recall = ax_recall.imshow(best(scores, criterion, dim=1), cmap=score_map, norm=divnorm)
    texts = annotate_heatmap(im_recall, data=match(scores, criterion, threshold, dim=1), valfmt="bool_char", threshold=threshold, criterion=criterion)

    # Turn spines off and create white grid for precision bar
    ax_precision.spines[:].set_visible(False)
    ax_precision.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    ax_precision.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax_precision.tick_params(which="minor", bottom=False, left=False)
    ax_precision.text(-0.6, 0, f"precision: {avg_precision:.2f}", ha='right', va='center', weight='bold')

    # Turn spines off and create white grid for recall bar
    ax_recall.spines[:].set_visible(False)
    ax_recall.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    ax_recall.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax_recall.tick_params(which="minor", bottom=False, left=False)
    ax_recall.text(0, -0.6, f"recall: {avg_recall:.2f}", rotation=-30, ha="right", rotation_mode="anchor", weight='bold')

    # Show only the text for f1 score
    ax_f1.spines[:].set_visible(False)
    ax_f1.axes.xaxis.set_visible(False)
    ax_f1.axes.yaxis.set_visible(False)
    ax_f1.text(0, 0, f"F1 score\n{f1:.2f}", ha='center', va='center', weight='bold')

    fig.tight_layout()
    return im_heatmap

##
##  Plotting a dialogue
##

# Constants for the vertical space per line and turn. Shloud be adjusted depending on fontstyle
PER_LINE = 0.22 # inch
PER_TURN = 0.15 # inch

def save_dialogue_fig(wrapped_turns, title, savepath):

    # Setup figure
    total_lines = sum([len(t[1]) for t in wrapped_turns])
    fig_height = 0.5 + len(wrapped_turns) * PER_TURN + total_lines * PER_LINE
    fig, ax = plt.subplots(figsize=(6, fig_height))
    fig.patch.set_facecolor('ghostwhite')

    # Determine triangle coordinates based on figure size
    triangle = np.array([[0.02, -0.05/fig_height], [0.05, -0.25/fig_height], [0.12, -0.25/fig_height]])

    ypos = 0.2 / fig_height 
    for i, (speaker, wrapped_turn) in enumerate(wrapped_turns):

        # Set alignment alternating left or right
        alignment = {"you": 'left', "me": 'right', "sessionbreak": 'center'}[speaker]
        xpos = {"you": 0.05, "me": 0.95, "sessionbreak": 0.5}[speaker]
        bbox_style = dict(
            boxstyle="round", 
            fc={"you": 'antiquewhite', "me": 'antiquewhite', "sessionbreak": 'lightsteelblue'}[speaker], 
            ec='tab:blue'
        )

        # Different style for last utterance
        if i == len(wrapped_turns) - 1:
            bbox_style['fc'] = 'floralwhite'
            bbox_style['linestyle'] = '--'

        # Plot the text
        text = ax.text(xpos, ypos, '\n'.join(wrapped_turn), 
            horizontalalignment=alignment,
            verticalalignment='top',
            wrap=True, 
            multialignment=alignment,
            bbox=bbox_style
        )

        # Increase ypos, for next utterance, depending on number of lines in current turn
        ypos += PER_TURN / fig_height + PER_LINE / fig_height * len(wrapped_turn)

        # Plot triangle below utterance, pointing left or right depending on speaker
        if speaker == "you":
            ax.add_patch(matplotlib.patches.Polygon(np.array([[0, ypos]]) + triangle))
        elif speaker == "me":
            ax.add_patch(matplotlib.patches.Polygon(np.array([[1, ypos]]) + triangle * np.array([[-1, 1]])))

    # Final formatting
    ax.invert_yaxis()
    ax.set_title(title)
    plt.axis('off')
    plt.savefig(savepath, pad_inches=0.2, bbox_inches='tight')
    plt.close(fig)

    return fig

if __name__ == "__main__":
    # matches = torch.rand((4,6))
    # targets = ["A" * 35, "b"*20, 'c' * 40, 'd'* 15]
    # predictions = [str(i) * 25 for i in range(6)]
    # threshold = 0.8
    # criterion = lambda x, threshold: x >= threshold
    # im = plot_heatmap(matches, threshold, criterion, targets, predictions, title="this is a title")
    # plt.show()
    # im.figure.savefig("test.jpg")
    pass
