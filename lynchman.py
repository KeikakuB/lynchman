#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import json
import sys
import collections

from mutagen.oggvorbis import OggVorbis

SONG_DIFFICULTIES=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
SONG_EXTENSION=".json"
SONGFILE_EXTENSION=".ogg"

OPERATIONS=["multi", "single", "compare"]

from enum import Enum

class NoteType(Enum):
    NONE = 1
    ALL = 2
    NORMAL = 3
    LEFT = 4
    RIGHT = 5
    BOMB = 6

class SongCollection:
    def __init__(self, root_directory, difficulty=None, text_filter=None, max_count=None):
        self.difficulty = difficulty
        self.text_filter = text_filter
        self.max_count = max_count
        songs = []
        for ident_dir in os.listdir(root_directory):
            if self.max_count and len(songs) >= max_count:
                break
            root = os.path.join(root_directory, ident_dir)
            tmp_songs = []
            song_file = None
            song_ident = os.path.basename(ident_dir)
            if '-' not in song_ident:
               continue
            if os.path.isdir(root):
                for name_dir in os.listdir(root):
                    root = os.path.join(root, name_dir)
                    song_name = os.path.basename(name_dir)
                    if self.text_filter and self.text_filter not in song_name:
                        continue
                    if os.path.isdir(root):
                        for item in os.listdir(root):
                            is_song = False
                            filepath = os.path.join(root, item)
                            (name, ext) = os.path.splitext(item)
                            if os.path.isfile(filepath) and ext == SONGFILE_EXTENSION:
                                song_file = filepath
                            if os.path.isfile(filepath) and ext == SONG_EXTENSION:
                                if self.difficulty:
                                    is_song = name == self.difficulty
                                else:
                                    is_song = name in SONG_DIFFICULTIES
                            if is_song:
                                tmp_songs.append((song_ident, song_name, name, filepath))
            for s in tmp_songs:
                songs.append(Song(s[0], s[1], s[2], s[3], song_file))
        self._songs = songs
        self._number_of_songs = len(self._songs)

    def get_songs(self):
        return self._songs

    def get_number_of_songs(self):
        return self._number_of_songs

    def get_notes(self, note_type=NoteType.ALL):
        notes_by_songs = [s.get_notes(note_type) for s in self._songs]
        notes = [s for song in notes_by_songs for s in song]
        return notes

    def get_beats_per_minute(self):
        return sum([s.get_beats_per_minute() for s in self._songs]) / self.get_number_of_songs()

    def get_beats_per_minute_std(self):
        return np.std([s.get_beats_per_minute() for s in self._songs])

    def get_beats_per_bar(self):
        return sum([s.get_beats_per_bar() for s in self._songs]) / self.get_number_of_songs()

    def get_beats_per_bar_std(self):
        return np.std([s.get_beats_per_bar() for s in self._songs])

    def get_note_jump_speed(self):
        return sum([s.get_note_jump_speed() for s in self._songs]) / self.get_number_of_songs()

    def get_note_jump_speed_std(self):
        return np.std([s.get_note_jump_speed() for s in self._songs])

    def get_shuffle(self):
        return sum([s.get_shuffle() for s in self._songs]) / self.get_number_of_songs()

    def get_shuffle_std(self):
        return np.std([s.get_shuffle() for s in self._songs])

    def get_shuffle_period(self):
        return sum([s.get_shuffle_period() for s in self._songs]) / self.get_number_of_songs()

    def get_shuffle_period_std(self):
        return np.std([s.get_shuffle_period() for s in self._songs])

    def get_average_duration_in_seconds_between_notes(self):
        return sum([s.get_average_duration_in_seconds_between_notes() for s in self._songs]) / self.get_number_of_songs()

    def get_average_duration_in_seconds_between_notes_std(self):
        return np.std([s.get_average_duration_in_seconds_between_notes() for s in self._songs])

    def get_left_right_lean_fraction(self):
        return sum([s.get_left_right_lean_fraction() for s in self._songs]) / self.get_number_of_songs()

    def get_left_right_lean_fraction_std(self):
        return np.std([s.get_left_right_lean_fraction() for s in self._songs])

    def get_duration(self):
        return sum([s.get_duration() for s in self._songs]) / self.get_number_of_songs()

    def get_duration_std(self):
        return np.std([s.get_duration() for s in self._songs])

class Song:
    def __init__(self, ident, name, difficulty, json_filepath, song_filepath):
        self.ident = ident
        self.name = name
        self.difficulty = difficulty
        with open(json_filepath) as f:
            self._data = json.load(f)
        self.audio_info = OggVorbis(song_filepath)
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

    def get_version(self):
        return self._data["_version"]

    def get_duration(self):
        return self.audio_info.info.length

    def get_beats_per_minute(self):
        return self._data["_beatsPerMinute"]

    def get_beats_per_bar(self):
        return self._data["_beatsPerBar"]

    def get_note_jump_speed(self):
        return self._data["_noteJumpSpeed"]

    def get_shuffle(self):
        return self._data["_shuffle"]

    def get_shuffle_period(self):
        return self._data["_shufflePeriod"]

    def get_average_duration_in_seconds_between_notes(self):
        all_normal_notes = self.get_notes(NoteType.NORMAL)
        diffs = []
        MAX_DURATION_IN_SECONDS_BETWEEN_NOTES = 4.0
        # compute the average time between notes
        for i in range(0, len(all_normal_notes) - 1):
            t1 = all_normal_notes[i]["_time"]
            t2 = all_normal_notes[i + 1]["_time"]
            abs_diff = abs(t2 - t1)
            if abs_diff < MAX_DURATION_IN_SECONDS_BETWEEN_NOTES:
                diffs.append(abs_diff)

        if len(diffs) == 0:
            return MAX_DURATION_IN_SECONDS_BETWEEN_NOTES
        return sum(diffs) / len(diffs)

    def get_left_right_lean_fraction(self):
        """ A positive number mean that the song has more right than left notes, a negative number means the opposite and zero means that there are an equal number of both. """
        n_left_notes = len(self.get_notes(NoteType.LEFT))
        n_right_notes = len(self.get_notes(NoteType.RIGHT))
        n_normal_notes = n_left_notes + n_right_notes
        return (n_right_notes - n_left_notes) / n_normal_notes

def get_basic_data_as_text(song):
    lines = []

    lines.append("beatsPerMinute: {:.2f}".format(song.get_beats_per_minute()))
    lines.append("beatsPerBar: {:.2f}".format(song.get_beats_per_bar()))
    lines.append("noteJumpSpeed: {:.2f}".format(song.get_note_jump_speed()))
    lines.append("shuffle: {:.2f}".format(song.get_shuffle()))
    lines.append("shufflePeriod: {:.2f}".format(song.get_shuffle_period()))

    n_left_notes = len(song.get_notes(NoteType.LEFT))
    n_right_notes = len(song.get_notes(NoteType.RIGHT))

    n_normal_notes = n_left_notes + n_right_notes
    lines.append("total normal notes: {}".format(n_normal_notes))
    lines.append("total bombs: {}".format(len(song.get_notes(NoteType.BOMB))))

    lines.append("notes count (left,right): ({}, {})".format(n_left_notes, n_right_notes))

    left_note_fraction = n_left_notes / n_normal_notes
    right_note_fraction = n_right_notes / n_normal_notes

    lines.append("notes leaning (left-, right+): {:.2f}".format(song.get_left_right_lean_fraction()))

    lines.append("average difference in seconds between notes: {:.2f}".format(song.get_average_duration_in_seconds_between_notes()))

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

    if len(notes) > 0:
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
    plt.text(0.05,0.9, "Percentage of cut directions", transform=fig.transFigure, size=14)
    plt.text(0.05,0.05, "eg. N means you have to cut the block from the top", transform=fig.transFigure, size=6)

    s = 0.3

    shifts = [(0, -s), (0, s), (s, 0), (-s, 0), (s, -s), (-s, -s), (s, s), (-s, s), (0, 0)]
    center_point_coord = 0.5
    direction_scale = 0.7
    direction_names = ["S", "N", "E", "W", "SE", "SW", "NE", "NW"]

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
        x = center_point_coord + shift[0]
        y = center_point_coord + shift[1]
        plt.text(x, y, "{:.2f}".format(cut_percentages[i]), transform=fig.transFigure, ha="center", family='sans-serif', size=14)
        if i < len(shifts) - 1:
            plt.text(center_point_coord + shift[0] * direction_scale, center_point_coord + shift[1] * direction_scale, direction_names[i], transform=fig.transFigure, ha="center", family='sans-serif', size=14)

    plt.axis('off')

def build_basic_text(text):
    fig = plt.figure()
    plt.axis('off')
    plt.text(0.05,0.05,text, transform=fig.transFigure, size=14)

def save_bar_charts_pdf(pdf_filepath, song_collections):
    with PdfPages(pdf_filepath) as pdf:
        build_bar_chart(song_collections, 'Number of songs', lambda c : c.get_number_of_songs())
        pdf.savefig()
        plt.close()
        build_bar_chart(song_collections, 'Duration (s)', lambda c : c.get_duration(), lambda c : c.get_duration_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(song_collections, 'Average Duration between Notes (s)', lambda c : c.get_average_duration_in_seconds_between_notes(), lambda c : c.get_average_duration_in_seconds_between_notes_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(song_collections, 'Left/Right(-/+) Lean', lambda c : c.get_left_right_lean_fraction(), lambda c : c.get_left_right_lean_fraction_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(song_collections, 'Note Jump Speed', lambda c : c.get_note_jump_speed(), lambda c : c.get_note_jump_speed_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(song_collections, 'BMP', lambda c : c.get_beats_per_minute(), lambda c : c.get_beats_per_minute_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(song_collections, 'BMB', lambda c : c.get_beats_per_bar(), lambda c : c.get_beats_per_bar_std())
        pdf.savefig()
        plt.close()
        # build_bar_chart(song_collections, 'Shuffle', lambda c : c.get_shuffle(), lambda c : c.get_shuffle_std())
        # pdf.savefig()
        # plt.close()
        # build_bar_chart(song_collections, 'Shuffle period', lambda c : c.get_shuffle_period(), lambda c : c.get_shuffle_period_std())
        # pdf.savefig()
        # plt.close()

def build_bar_chart(song_collections, value_name, fn_val, fn_std=None, color='b'):
    n_difficulties = len(song_collections)

    values = tuple([fn_val(c) for c in song_collections])
    std_values = None
    if fn_std:
        std_values = tuple([fn_std(c) for c in song_collections])

    fig, ax = plt.subplots()

    index = np.arange(n_difficulties)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index + bar_width / 2, values, bar_width,
                    alpha=opacity, color=color,
                    yerr=std_values, error_kw=error_config)

    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for i in range(len(rects1)):
        rect = rects1[i]
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = rect.get_width()
        height = rect.get_height()

        # Center the text vertically in the bar
        xloc = rect.get_x() + width / 2
        yloc = rect.get_y() + height / 2
        label = ax.text(xloc, yloc, "{:.2f}".format(values[i]), horizontalalignment='center',
                         verticalalignment='center', color='black', weight='bold',
                         clip_on=True, size=7)
        rect_labels.append(label)

    ax.set_xlabel('Difficulty')
    ax.set_ylabel(value_name)
    ax.set_title("{} by Difficulty".format(value_name))
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tuple([c.difficulty for c in song_collections]))

    fig.tight_layout()

def save_pdf(pdf_filepath, song):
    with PdfPages(pdf_filepath) as pdf:
        build_basic_text(get_basic_data_as_text(song))
        pdf.savefig()
        plt.close()
        all_normal_notes = song.get_notes(NoteType.NORMAL)
        build_note_heatmap(all_normal_notes)
        plt.text(0, 2.75, "Total normal notes: {}".format(len(all_normal_notes)))
        pdf.savefig()
        plt.close()
        all_bomb_notes = song.get_notes(NoteType.BOMB)
        build_note_heatmap(all_bomb_notes, "Reds")
        plt.text(0, 2.75, "Total bombs: {}".format(len(all_bomb_notes)))
        pdf.savefig()
        plt.close()
        all_notes = song.get_notes()
        build_histogram(all_notes)
        pdf.savefig()
        plt.close()
        build_cut_directions_drawing(all_normal_notes)
        pdf.savefig()
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("python3 banalyze.py JSON_FILEPATH")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--path', help="Custom songs folder path for analyzing many custom songs.")
    parser.add_argument('--difficulty', type=str, choices=SONG_DIFFICULTIES, default=None, help='Difficulty of songs to analyze, if not set then analyze all difficulties')
    parser.add_argument('--text_filter', default=None, help='text to use to filter songs')
    parser.add_argument('--max_count', type=int, default=None, help="Numbers of songs to analyse, -1 then no maximum.")
    parser.add_argument('--output_filepath', type=str, default="OUTPUT", help='Filename to output to when performing certain operations.')
    parser.add_argument('--operations', nargs='+', choices=OPERATIONS, default=[OPERATIONS[0]], help='')

    args = parser.parse_args()

    is_operation_handled = False
    if OPERATIONS[0] in args.operations:
        is_operation_handled = True
        song_collection = SongCollection(args.path, difficulty=args.difficulty, text_filter=args.text_filter, max_count=args.max_count)
        save_pdf("{}.pdf".format(args.output_filepath), song_collection)
        song_descriptors = ["{} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in song_collection.get_songs()]
        with open("{}.txt".format(args.output_filepath), 'wt', encoding='utf-8') as f:
            f.write('\n'.join(song_descriptors))
    if OPERATIONS[1] in args.operations:
        is_operation_handled = True
        song_collection = SongCollection(args.path, difficulty=args.difficulty, text_filter=args.text_filter, max_count=args.max_count)
        for s in song_collection.get_songs():
            save_pdf("{}_{}_{}_{}.pdf".format(args.output_filepath, s.ident, s.name, s.difficulty), s)
    if OPERATIONS[2] in args.operations:
        is_operation_handled = True
        collections = []
        lines = []
        for d in SONG_DIFFICULTIES:
            song_collection = SongCollection(args.path, difficulty=d, max_count=args.max_count, text_filter=args.text_filter)
            lines.append("{} - {} songs".format(d, song_collection.get_number_of_songs()))
            collections.append(song_collection)
            song_descriptors = ["  {} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in song_collection.get_songs()]
            lines.extend(song_descriptors)
        with open("{}.txt".format(args.output_filepath), 'wt', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        save_bar_charts_pdf("{}.pdf".format(args.output_filepath), collections)

    if not is_operation_handled:
        print("Unhandled operation type, error in code")
        sys.exit(1)

if __name__ == "__main__":
    main()
