#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import sys
import collections

from enum import Enum
class NoteType(Enum):
    NONE = 1
    ALL = 2
    NORMAL = 3
    LEFT = 4
    RIGHT = 5
    BOMB = 6

class Song:
    def __init__(self, ident, name, json_filepath):
        self.ident = ident
        self.name = name
        with open(json_filepath) as f:
            self._data = json.load(f)
        self._events = self._data["_events"]
        self._notes = self._data["_notes"]
        self._notes_normal = self._get_notes(NoteType.NORMAL)
        self._notes_left = self._get_notes(NoteType.LEFT)
        self._notes_right = self._get_notes(NoteType.RIGHT)
        self._notes_bomb = self._get_notes(NoteType.BOMB)

    def get_notes(self, note_type=NoteType.ALL):
        if note_type == NoteType.ALL:
            return self._notes
        elif note_type == NoteType.NORMAL:
            return self._notes_normal
        elif note_type == NoteType.LEFT:
            return self._notes_left
        elif note_type == NoteType.RIGHT:
            return self._notes_right
        elif note_type == NoteType.BOMB:
            return self._notes_bomb

    def _get_notes(self, note_type=NoteType.ALL):
        # note_type: all, normal, left, right, bombs
        if note_type is NoteType.ALL:
            return self._notes
        ls = []
        for n in self._notes:
            type_value = n["_type"]
            is_included = False
            if type_value == 0 or type_value == 1:
                if note_type is NoteType.NORMAL:
                    is_included = True
                elif note_type is NoteType.LEFT and type_value == 0:
                    is_included = True
                elif note_type is NoteType.RIGHT and type_value == 1:
                    is_included = True
            if note_type is NoteType.BOMB and type_value == 3:
                is_included = True
            if is_included:
                ls.append(n)
        return ls

    def log_basic_data(self):
        lines = []
        header_data_names = ["_version", "_beatsPerMinute", "_beatsPerBar", "_noteJumpSpeed", "_shuffle", "_shufflePeriod"]
        for n in header_data_names:
            lines.append("{}: {}".format(n[1:], self._data[n]))

        left_note_count = len(self.get_notes(NoteType.LEFT))
        right_note_count = len(self.get_notes(NoteType.RIGHT))

        total_normal_notes = left_note_count + right_note_count
        lines.append("total normal notes: {}".format(total_normal_notes))
        lines.append("total bombs: {}".format(len(self.get_notes(NoteType.BOMB))))

        lines.append("notes count (left,right): ({}, {})".format(left_note_count, right_note_count))

        left_note_fraction = left_note_count / total_normal_notes
        right_note_fraction = right_note_count / total_normal_notes

        lines.append("notes leaning (left+, right-): {:.2f}".format(left_note_fraction - right_note_fraction))
        return '\n'.join(lines)

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
    ax.set_ylabel('Number of notes')
    ax.set_title(r'Notes over time')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

def build_cut_directions_drawing(notes):
    fig, ax = plt.subplots()

    s = 0.25

    shifts = [(0, -s), (0, s), (s, 0), (-s, 0), (s, -s), (-s, -s), (s, s), (-s, s), (0, 0)]

    cut_direction_count = collections.Counter()

    for i in range(0, 9):
        cut_direction_count[i] = 0

    for n in notes:
        cut_direction_count[n["_cutDirection"]] += 1

    total_cuts = sum(cut_direction_count.values())
    cut_percentages = []
    for k in sorted(cut_direction_count.keys()):
        cut_count = cut_direction_count[k]
        cut_percentage = cut_count / total_cuts
        cut_percentages.append(cut_percentage)

    for i in range(0, len(shifts)):
        shift = shifts[i]
        x = 0.5 + shift[0]
        y = 0.5 + shift[1]
        plt.text(x, y, "{:.2f}".format(cut_percentages[i]), transform=fig.transFigure, ha="center", family='sans-serif', size=14)

    plt.axis('off')

def build_basic_text(text):
    fig = plt.figure()
    plt.axis('off')
    plt.text(0.05,0.05,text, transform=fig.transFigure, size=24)

def main():
    if len(sys.argv) < 2:
        print("python3 banalyze.py JSON_FILEPATH")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--path', help="Custom songs folder path for analyzing many custom songs.")
    parser.add_argument('--limit', type=int, default=None, help='numbers of songs to analyse, -1 then no limit')
    parser.add_argument('--difficulty', type=str, choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"], default="Expert", help='Difficulty of songs to analyze')
    parser.add_argument('--filter', default=None, help='text to use to filter songs')
    parser.add_argument('--operation', choices=['cumulative', 'single'], default='cumulative', help='Print more data')

    args = parser.parse_args()

    songs = []
    for ident_dir in os.listdir(args.path):
        if args.limit and len(songs) > args.limit:
            break
        root = os.path.join(args.path, ident_dir)
        song_ident = os.path.basename(ident_dir)
        if '-' not in song_ident:
           continue
        if os.path.isdir(root):
            for name_dir in os.listdir(root):
                root = os.path.join(root, name_dir)
                song_name = os.path.basename(name_dir)
                if args.filter and args.filter not in song_name:
                    continue
                if os.path.isdir(root):
                    for item in os.listdir(root):
                        if os.path.isfile(os.path.join(root, item)) and "{}.json".format(args.difficulty) in item:
                            song_json_filepath = os.path.join(root, item)
                            print("{} _ {}".format(song_ident, song_name))
                            songs.append(Song(song_ident, song_name, song_json_filepath))

    if args.operation == 'cumulative':
        with PdfPages("out/cumul.pdf") as pdf:
            notes_by_songs = [s.get_notes(NoteType.NORMAL) for s in songs]
            all_normal_notes = sorted([s for song in notes_by_songs for s in song], key = lambda x : x["_time"])
            build_note_heatmap(all_normal_notes)
            plt.text(0, 2.75, "total notes: {}".format(len(all_normal_notes)))
            pdf.savefig()
            plt.close()
            bombs_by_songs = [s.get_notes(NoteType.BOMB) for s in songs]
            all_bombs_notes = sorted([s for song in bombs_by_songs for s in song], key = lambda x : x["_time"])
            build_note_heatmap(all_bombs_notes, "Reds")
            plt.text(0, 2.75, "total bombs: {}".format(len(all_bombs_notes)))
            pdf.savefig()
            plt.close()
            all_notes_by_songs = [s.get_notes() for s in songs]
            all_notes = sorted([s for song in all_notes_by_songs for s in song], key = lambda x : x["_time"])
            build_histogram(all_notes)
            pdf.savefig()
            plt.close()
            build_cut_directions_drawing(all_normal_notes)
            pdf.savefig()
            plt.close()
    elif args.operation == 'single':
        for s in songs:
            with PdfPages("out/{}_{}.pdf".format(s.ident, s.name)) as pdf:
                build_basic_text(s.log_basic_data())
                pdf.savefig()
                plt.close()
                build_note_heatmap(s.get_notes(NoteType.NORMAL))
                pdf.savefig()
                plt.close()
                build_note_heatmap(s.get_notes(NoteType.BOMB), "Reds")
                pdf.savefig()
                plt.close()
                build_histogram(s.get_notes())
                pdf.savefig()
                plt.close()
                build_cut_directions_drawing(s.get_notes(NoteType.NORMAL))
                pdf.savefig()
                plt.close()
    else:
        print("Unhandled operation type, error in code")
        sys.exit(1)


if __name__ == "__main__":
    main()
