#!/usr/bin/env python

import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import sys
import collections

def build_note_heatmap(notes, cmap="YlGn"):
    notes_count = collections.Counter()

    for n in notes:
        position = (n["_lineIndex"], n["_lineLayer"])
        notes_count[position] += 1

    line_layers = ["L2", "L1", "L0"]
    line_indices = ["I0", "I1", "I2", "I3"]

    data = np.array([[notes_count[(0, 2)], notes_count[(1, 2)], notes_count[(2, 2)], notes_count[(3, 2)]],
                        [notes_count[(0, 1)], notes_count[(1, 1)], notes_count[(2, 1)], notes_count[(3, 1)]],
                        [notes_count[(0, 0)], notes_count[(1, 0)], notes_count[(2, 0)], notes_count[(3, 0)]]])

    data = data / len(notes)

    fig, ax = plt.subplots()

    (im, cbar) = heatmap(data, line_layers, line_indices, ax=ax, cmap=cmap, cbarlabel="Notes heatmap by grid position")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def build_histogram(notes, n_bins=60):
    times = [n["_time"] for n in notes]

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(times, n_bins)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of Note')
    ax.set_title(r'Histogram of Note over time')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("python3 banalyze.py JSON_FILEPATH")
        sys.exit(1)
    input_filename = sys.argv[1]
    with open(input_filename) as f:
        data = json.load(f)
    header_data_names = ["_version", "_beatsPerMinute", "_beatsPerBar", "_noteJumpSpeed", "_shuffle", "_shufflePeriod"]
    print("HEADERS")
    for n in header_data_names:
        print("{}: {}".format(n[1:], data[n]))

    notes = data["_notes"]

    left_note_count = len([n for n in notes if n["_type"] == 0])
    right_note_count = len([n for n in notes if n["_type"] == 1])

    total_normal_notes = left_note_count + right_note_count
    print("total normal notes: {}".format(total_normal_notes))

    print("notes count (left,right): ({}, {})".format(left_note_count, right_note_count))

    left_note_percentage = len([n for n in notes if n["_type"] == 0]) / total_normal_notes
    right_note_percentage = len([n for n in notes if n["_type"] == 1]) / total_normal_notes

    print("notes percentage (left,right): ({}, {})".format(left_note_percentage, right_note_percentage))

    cut_direction_count = collections.Counter()

    for n in notes:
        cut_direction_count[n["_cutDirection"]] += 1

    total_cuts = sum(cut_direction_count.values())
    cut_percentages = []
    for k in sorted(cut_direction_count.keys()):
        cut_count = cut_direction_count[k]
        cut_percentage = cut_count / total_cuts
        cut_percentages.append(cut_percentage)
        print("{}: {} ({:.2f})".format(k, cut_count, cut_percentage))

    print(
"""
       {:.2f}
      ------
     |      |
{:.2f} | {:.2f} | {:.2f}
     |      |
      ------
       {:.2f}

       -
      / \\
{:.2f} /   \ {:.2f}
    /     \\
    \     /
{:.2f} \   / {:.2f}
      \ /
       -
""".format(cut_percentages[1],cut_percentages[3],cut_percentages[8],cut_percentages[2],cut_percentages[1],cut_percentages[7],cut_percentages[6],cut_percentages[5],cut_percentages[4]))

    with PdfPages('out/out.pdf') as pdf:
        normal_notes = [n for n in notes if n["_type"] == 0 or n["_type"] == 1]
        build_note_heatmap(normal_notes)
        pdf.savefig()
        plt.close()
        bomb_notes = [n for n in notes if n["_type"] == 3]
        build_note_heatmap(bomb_notes, "Reds")
        pdf.savefig()
        plt.close()
        build_histogram(notes)
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    main()
