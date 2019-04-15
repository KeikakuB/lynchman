#!/usr/bin/env python

import os
import random
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
import librosa

from mutagen.oggvorbis import OggVorbis
import math
from operator import itemgetter, attrgetter

N_HANDS=2
N_LINE_INDEX=4
N_LINE_LAYER=3
N_CUT_DIRECTIONS=9

MAP_DIFFICULTIES=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
MAP_EXTENSION=".json"
SONG_EXTENSION=".ogg"

OPERATIONS=["multi", "single", "compare", "generate"]


from enum import Enum

Block = namedtuple('Block', ['type', 'coords', 'cut_direction'])

def make_block_from_note(n):
    return Block(n["_type"], (n["_lineIndex"], n["_lineLayer"]), n["_cutDirection"])

class JsonNoteType(Enum):
    """JSON values for the `_type` field of a note."""
    LEFT = 0
    RIGHT = 1
    BOMB = 3

class JsonNoteCutDirection(Enum):
    """JSON values for the `_cutDirection` field of a note."""
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    SOUTH_EAST = 4
    SOUTH_WEST = 5
    NORTH_EAST = 6
    NORTH_WEST = 7
    NONE = 8

class NoteGroupType(Enum):
    """Enum used for picking which type of notes we're looking at."""
    NONE = 1
    ALL = 2
    NORMAL = 3
    LEFT = 4
    RIGHT = 5
    BOMB = 6

class Song:
    def __init__(self, audio_filepath):
        self._audio_filepath = audio_filepath
        # Load the audio as a waveform `y` and the sampling rate as `sr`
        self._y, self._sr = librosa.load(audio_filepath)

        # Run the default beat tracker
        self._tempo, self._beat_frames = librosa.beat.beat_track(y=self._y, sr=self._sr)

        # Convert the frame indices of beat events into timestamps
        self._beat_times = librosa.frames_to_time(self._beat_frames, sr=self._sr)

        self._beats_per_minute = round(float("{:.2f}".format(self._tempo)))

    def get_beat_times(self):
        return self._beat_times

    def get_beats_per_minute(self):
        return self._beats_per_minute

    def _generate_map(self, output_directory, difficulty_name, map_generator):
        map_generator.generate()

        in_events = map_generator.get_events()
        in_notes = map_generator.get_notes()
        in_obstacles = map_generator.get_obstacles()


        data = {}
        data["_version"] = "1.5.0"
        data["_beatsPerMinute"] = self.get_beats_per_minute()
        data["_beatsPerBar"] = 16
        data["_noteJumpSpeed"] = 10
        data["_shuffle"] = 0
        data["_shufflePeriod"] = 0.5
        data["_time"] = 0
        data["_events"] = in_events
        data["_notes"] = in_notes
        data["_obstacles"] = in_obstacles
        data["_bookmarks"] = []

        with open(os.path.join(output_directory, "{}.json".format(difficulty_name)), 'w') as outfile:
            json.dump(data, outfile)

    def generate_maps(self, output_directory, map_generators):
        for (name, _, generator) in map_generators:
            self._generate_map(output_directory, name, generator)

        audio_filename = os.path.basename(self._audio_filepath)
        (audio_name, _) = os.path.splitext(audio_filename)
        info_data = {}
        info_data["authorName"] = "KeikakuB"
        info_data["beatsPerMinute"] = self.get_beats_per_minute()
        info_data["coverImagePath"] = "cover.jpg"
        difficulty_levels = []
        for (name, rank, generator) in map_generators:
            difficulty_levels.append({
                "audioPath": audio_filename,
                "difficulty": name,
                "difficultyRank": rank,
                "jsonPath": "{}{}".format(name, MAP_EXTENSION),
                "offset": 0,
                "oldOffset": 0
            }
            )
        info_data["difficultyLevels"] = difficulty_levels
        info_data["environmentName"] =  "DefaultEnvironment"
        info_data["previewDuration"] =  10
        info_data["previewStartTime"] =  12
        info_data["songName"] =  audio_name
        info_data["songSubName"] =  ""

        with open(os.path.join(output_directory, "info.json"), 'w') as outfile:
            json.dump(info_data, outfile)

class MapGeneratorStrategy:
    """Abstract map generator strategy class, meant to be subclasses by concrete implementations.

       Children must implement a `_generate()` method in which they call the various `_add_*()`
         methods to create the map.
    """
    def __init__(self, song):
        self._song = song
        self._notes = []
        self._obstacles = []
        self._events = []


    def generate(self):
        self._notes = []
        self._obstacles = []
        self._events = []
        self._generate()

    def get_notes(self):
        return self._notes

    def get_obstacles(self):
        return self._obstacles

    def get_events(self):
        return self._events

    def _add_note(self, time, block):
        self._notes.append({
            "_time": float("{:.16f}".format(time)),
            "_lineIndex": block.coords[0],
            "_lineLayer": block.coords[1],
            "_type": block.type,
            "_cutDirection": block.cut_direction
        }
        )
    # def _add_event()
    # def _add_obstacle()

class MapGeneratorBeatStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which the same note is placed on every beat."""
    def __init__(self, song):
        MapGeneratorStrategy.__init__(self, song)

    def _generate(self):
        beat_times = self._song.get_beat_times()
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            self._add_note(float("{:.16f}".format(b)), Block(JsonNoteType.RIGHT.value, (2, 0) , N_CUT_DIRECTIONS - 1))

class MapGeneratorRandomStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which notes are placed completely randomly on every beat."""
    def __init__(self, song):
        MapGeneratorStrategy.__init__(self, song)

    def _generate(self):
        beat_times = self._song.get_beat_times()
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            hand = random.randrange(N_HANDS)
            note_type = JsonNoteType.RIGHT
            if hand == JsonNoteType.LEFT.value:
                note_type = JsonNoteType.LEFT
            line_index = random.randrange(N_LINE_INDEX)
            line_layer = random.randrange(N_LINE_LAYER)
            cut_direction = random.randrange(N_CUT_DIRECTIONS)
            self._add_note(float("{:.16f}".format(b)), Block(note_type.value, (line_index, line_layer), cut_direction))

class MapGeneratorWeightedRandomStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which notes are placed randomly on every beat
        with the probabilities of each note being determined by its representation in
        the given `map_collection`."""

    def __init__(self, song, map_collection):
        MapGeneratorStrategy.__init__(self, song)
        self._map_collection = map_collection

    def _generate(self):
        blocks = []
        for hand in range(N_HANDS):
            for index in range(N_LINE_INDEX):
                for layer in range(N_LINE_LAYER):
                    blocks.append((hand, index, layer))

        # Count the number of notes with each cut direction based on the notes location on the grid
        n_cut_directions_by_grid_position = {}
        for (hand, index, layer) in blocks:
            counter = collections.Counter()
            for c in range(N_CUT_DIRECTIONS):
                counter[c] = 0
            n_cut_directions_by_grid_position[(hand, index, layer)] = counter
        for m in self._map_collection.get_maps():
            for n in m.get_notes(NoteGroupType.NORMAL):
                hand = n["_type"]
                line_index = n["_lineIndex"]
                line_layer = n["_lineLayer"]
                cutDirection = n["_cutDirection"]
                n_cut_directions_by_grid_position[(hand, line_index, line_layer)][cutDirection] += 1

        # Convert the counts into probabilities (from 0.0 to 1.0) using the total counts per grid position
        #  sorting them in descending order.
        n_notes = n_cut_directions_by_grid_position.copy()
        for (hand, index, layer) in blocks:
            d = dict(n_cut_directions_by_grid_position[(hand, index, layer)])
            n_total = sum(d.values())
            n_notes[(hand, index, layer)] = n_total
            ls = [(k, v / n_total) for (k,v) in d.items()]
            n_cut_directions_by_grid_position[(hand, index, layer)] = sorted(ls, key=lambda a :  -a[1])

        # Convert the probabilities into 'cumulative probabilities' such that we can get a random float
        #  and use it to pick a cut direction from the list.
        # print(repr(n_cut_directions_by_grid_position[(0,0)]))
        for (hand, index, layer) in blocks:
            ls = n_cut_directions_by_grid_position[(hand, index, layer)]
            current_probability = 0
            for t in range(len(ls)):
                (a, probability) = ls[t]
                current_probability += probability
                ls[t] = (current_probability, a)

        # print(repr(n_notes))
        # Perform similar steps for the notes based on the grid position
        #   eg. what's the probability of having a note in the bottom left corner vs. the top left etc.
        n_total_notes = sum(n_notes.values())
        for (hand, index, layer) in blocks:
            n_notes[(hand, index, layer)] = ((hand, index, layer), n_notes[(hand, index, layer)] / n_total_notes)

        n_notes = n_notes.values()
        n_notes = sorted(n_notes, key=lambda a :  -a[1])
        # print(repr(n_notes))

        current_probability = 0
        for t in range(len(n_notes)):
            (coords, probability) = n_notes[t]
            current_probability += probability
            n_notes[t] = (current_probability, coords)
        # print(repr(n_notes))

        def get_weighted_random_coords():
            random_value = random.random()
            for (v, coords) in n_notes:
                if random_value <= v:
                    return coords
            return n_notes[len(n_notes) - 1][1]

        def get_weighted_random_cut_direction(hand, line_index, line_layer):
            ls = n_cut_directions_by_grid_position[(hand, line_index, line_layer)]
            random_value = random.random()
            for (v, cut_direction) in ls:
                if random_value <= v:
                    return cut_direction
            return ls [len(ls) - 1][1]

        # print(repr(n_cut_directions_by_grid_position))
        # print(repr(n_cut_directions_by_grid_position[(0,0)]))

        beat_times = self._song.get_beat_times()
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            (hand, line_index, line_layer) = get_weighted_random_coords()
            note_type = JsonNoteType.RIGHT
            if hand == JsonNoteType.LEFT.value:
                note_type = JsonNoteType.LEFT
            cut_direction = get_weighted_random_cut_direction(hand, line_index, line_layer)
            self._add_note(float("{:.16f}".format(b)), Block(note_type.value, (line_index, line_layer), cut_direction))

class MapGeneratorMarkovChainsStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which notes are placed randomly on every beat
        based on Markov chains determined the given `map_collection`."""

    def __init__(self, song, map_collection):
        MapGeneratorStrategy.__init__(self, song)
        self._map_collection = map_collection

    def _generate(self):
        patterns_by_map = []
        time_pattern_tuples_by_map = []
        for m in self._map_collection.get_maps():
            notes = m.get_notes(NoteGroupType.NORMAL)
            # Group notes into 'patterns' if they occur at the same time
            patterns = []
            time_pattern_tuples = []
            current_pattern = [make_block_from_note(notes[0])]
            for i in range(len(notes) - 1):
                l = notes[i]
                r = notes[i + 1]
                # TODO play with the fuzzy value -> but 1e-8 seems to work well
                if math.isclose(l["_time"], r["_time"], rel_tol=1e-8):
                    # Append right block and continue
                    current_pattern.append(make_block_from_note(r))
                else:
                    # Sort `current_pattern` to ensure uniqueness
                    current_pattern = tuple(sorted(current_pattern, key=attrgetter('type', 'coords', 'cut_direction')))
                    time_pattern_tuples.append((l["_time"], current_pattern))
                    patterns.append(current_pattern)
                    current_pattern = [make_block_from_note(r)]
            time_pattern_tuples_by_map.append(time_pattern_tuples)
            patterns_by_map.append(patterns)

        # Get probability of all patterns to be able to pick one weighted randomly at start and maybe in between sequences
        patterns_count = collections.Counter([item for sublist in patterns_by_map for item in sublist])
        n_patterns = sum(patterns_count.values())

        pattern_probabilities = []
        for (pattern, n_pattern) in patterns_count.items():
            pattern_probabilities.append((n_pattern / n_patterns, pattern))

        pattern_probabilities = sorted(pattern_probabilities, key=itemgetter(0), reverse=True)
        # print(repr(pattern_probabilities))

        cumulative_probability = 0
        for i in range(len(pattern_probabilities)):
            (prob, pattern) = pattern_probabilities[i]
            cumulative_probability += prob
            pattern_probabilities[i] = [cumulative_probability, pattern]

        # for i in range(20):
        #     print(repr(pattern_probabilities[i]))
        # print(len(pattern_probabilities))

        # this code generates a map with each pattern on beat in descending order of probability
        # beat_times = self._song.get_beat_times()
        # for i in range(0, len(beat_times), 2):
        #     if i >= len(pattern_probabilities):
        #         break
        #     (_, pattern) = pattern_probabilities[i]
        #     b = beat_times[i]
        #     for block in pattern:
        #         self._add_note(float("{:.16f}".format(b)), block)


        # Markov chains
        pattern_adjacency_counts = {}
        for (_, c) in pattern_probabilities:
            pattern_adjacency_counts[c] = collections.Counter()

        # todo "two beats"
        allowed_delay_in_seconds = 2 * (60 / self._song.get_beats_per_minute())
        for patterns in time_pattern_tuples_by_map:
            for i in range(len(patterns) - 1):
                (t1, l) = patterns[i]
                (t2, r) = patterns[i + 1]
                if math.isclose(t1, t2, rel_tol=allowed_delay_in_seconds):
                    pattern_adjacency_counts[l][r] += 1

        pattern_adjaceny_probabilities = {}
        for (pattern, count) in pattern_adjacency_counts.items():
            total_count = sum(pattern_adjacency_counts[pattern].values())
            pattern_adjaceny_probabilities[pattern] = [(pattern_adjacency_counts[pattern][c] / total_count, c) for c in pattern_adjacency_counts[pattern].keys()]

        pattern_adjacency_probabilities_cumulative = {}
        for (pattern, ls) in pattern_adjaceny_probabilities.items():
            ls = sorted(ls, key=itemgetter(0), reverse=True)
            cumulative_probability = 0
            for k in range(len(ls)):
                (prob, c) = ls[k]
                cumulative_probability += prob
                ls[k] = (cumulative_probability, c)
            pattern_adjacency_probabilities_cumulative[pattern] = ls

        def get_pattern():
            random_value = random.random()
            for (prob, pattern) in pattern_probabilities:
                if random_value <= prob:
                    return pattern
            return pattern_probabilities[len(pattern_probabilities) - 1][1]

        def get_next_pattern(last_pattern):
            ls = pattern_adjacency_probabilities_cumulative[last_pattern]
            random_value = random.random()
            for (prob, p) in ls:
                if random_value <= prob:
                    return p
            return None

        # todo figure out why notes stop early (check beat_times)
        beat_times = self._song.get_beat_times()
        current_pattern = get_pattern()
        n_patterns_in_current_sequence = 0
        n_sequence_pattern_limit = 31
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            if not current_pattern:
                print("Resetting pattern sequence after {}".format(b))
                current_pattern = get_pattern()
                # skip a beat to allow the player to reposition if needed
                continue
            for block in current_pattern:
                self._add_note(float("{:.16f}".format(b)), block)
            n_patterns_in_current_sequence += 1
            if n_patterns_in_current_sequence >= n_sequence_pattern_limit:
                current_pattern = None
                n_patterns_in_current_sequence = 0
            else:
                current_pattern = get_next_pattern(current_pattern)

class MapCollection:
    """Collection of maps to be analyzed as a group."""
    def __init__(self, root_directory, difficulty=None, text_filter=None, max_count=None):
        self.difficulty = difficulty
        self.text_filter = text_filter
        self.max_count = max_count
        maps = []
        for ident_dir in os.listdir(root_directory):
            if self.max_count and len(maps) >= max_count:
                break
            root = os.path.join(root_directory, ident_dir)
            tmp_maps = []
            audio_filepath = None
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
                            is_map = False
                            filepath = os.path.join(root, item)
                            (filename, ext) = os.path.splitext(item)
                            if os.path.isfile(filepath) and ext == SONG_EXTENSION:
                                audio_filepath = filepath
                            if os.path.isfile(filepath) and ext == MAP_EXTENSION:
                                if self.difficulty:
                                    is_map = filename == self.difficulty
                                else:
                                    is_map = filename in MAP_DIFFICULTIES
                            if is_map:
                                tmp_maps.append((song_ident, song_name, filename, filepath))
            for s in tmp_maps:
                maps.append(Map(s[0], s[1], s[2], s[3], audio_filepath))
        self._maps = maps
        self._n_maps = len(self._maps)

    def get_maps(self):
        return self._maps

    def get_number_of_maps(self):
        return self._n_maps

    def get_notes(self, note_type=NoteGroupType.ALL):
        notes_by_map = [m.get_notes(note_type) for m in self._maps]
        notes = [m for _map in notes_by_map for m in _map]
        return notes

    def get_beats_per_minute(self):
        return sum([s.get_beats_per_minute() for s in self._maps]) / self.get_number_of_maps()

    def get_beats_per_minute_std(self):
        return np.std([s.get_beats_per_minute() for s in self._maps])

    def get_beats_per_bar(self):
        return sum([s.get_beats_per_bar() for s in self._maps]) / self.get_number_of_maps()

    def get_beats_per_bar_std(self):
        return np.std([s.get_beats_per_bar() for s in self._maps])

    def get_note_jump_speed(self):
        return sum([s.get_note_jump_speed() for s in self._maps]) / self.get_number_of_maps()

    def get_note_jump_speed_std(self):
        return np.std([s.get_note_jump_speed() for s in self._maps])

    def get_shuffle(self):
        return sum([s.get_shuffle() for s in self._maps]) / self.get_number_of_maps()

    def get_shuffle_std(self):
        return np.std([s.get_shuffle() for s in self._maps])

    def get_shuffle_period(self):
        return sum([s.get_shuffle_period() for s in self._maps]) / self.get_number_of_maps()

    def get_shuffle_period_std(self):
        return np.std([s.get_shuffle_period() for s in self._maps])

    def get_average_duration_in_seconds_between_notes(self):
        return sum([s.get_average_duration_in_seconds_between_notes() for s in self._maps]) / self.get_number_of_maps()

    def get_average_duration_in_seconds_between_notes_std(self):
        return np.std([s.get_average_duration_in_seconds_between_notes() for s in self._maps])

    def get_left_right_lean_fraction(self):
        return sum([s.get_left_right_lean_fraction() for s in self._maps]) / self.get_number_of_maps()

    def get_left_right_lean_fraction_std(self):
        return np.std([s.get_left_right_lean_fraction() for s in self._maps])

    def get_duration(self):
        return sum([s.get_duration() for s in self._maps]) / self.get_number_of_maps()

    def get_duration_std(self):
        return np.std([s.get_duration() for s in self._maps])

class Map:
    """Beat Saber map."""
    def __init__(self, ident, name, difficulty, map_filepath, audio_filepath):
        self.ident = ident
        self.name = name
        self.difficulty = difficulty
        with open(map_filepath) as f:
            self._data = json.load(f)
        self.audio_info = OggVorbis(audio_filepath)
        self._events = self._data["_events"]
        self._notes = self._data["_notes"]
        self._notes_normal = self._get_notes(NoteGroupType.NORMAL)
        self._notes_left = self._get_notes(NoteGroupType.LEFT)
        self._notes_right = self._get_notes(NoteGroupType.RIGHT)
        self._notes_bomb = self._get_notes(NoteGroupType.BOMB)

    def get_notes(self, note_type=NoteGroupType.ALL):
        if note_type == NoteGroupType.ALL:
            return self._notes
        elif note_type == NoteGroupType.NORMAL:
            return self._notes_normal
        elif note_type == NoteGroupType.LEFT:
            return self._notes_left
        elif note_type == NoteGroupType.RIGHT:
            return self._notes_right
        elif note_type == NoteGroupType.BOMB:
            return self._notes_bomb

    def _get_notes(self, note_type=NoteGroupType.ALL):
        # note_type: all, normal, left, right, bombs
        if note_type is NoteGroupType.ALL:
            return self._notes
        ls = []
        for n in self._notes:
            type_value = n["_type"]
            is_included = False
            if type_value == 0 or type_value == 1:
                if note_type is NoteGroupType.NORMAL:
                    is_included = True
                elif note_type is NoteGroupType.LEFT and type_value == 0:
                    is_included = True
                elif note_type is NoteGroupType.RIGHT and type_value == 1:
                    is_included = True
            if note_type is NoteGroupType.BOMB and type_value == 3:
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
        all_normal_notes = self.get_notes(NoteGroupType.NORMAL)
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
        """ A positive number mean that the map has more right than left notes, a negative number means the opposite and zero means that there are an equal number of both. """
        n_left_notes = len(self.get_notes(NoteGroupType.LEFT))
        n_right_notes = len(self.get_notes(NoteGroupType.RIGHT))
        n_normal_notes = n_left_notes + n_right_notes
        return (n_right_notes - n_left_notes) / n_normal_notes

def get_basic_data_as_text(_map):
    lines = []

    lines.append("duration (s): {:.2f}".format(_map.get_duration()))
    lines.append("beatsPerMinute: {:.2f}".format(_map.get_beats_per_minute()))
    lines.append("beatsPerBar: {:.2f}".format(_map.get_beats_per_bar()))
    lines.append("noteJumpSpeed: {:.2f}".format(_map.get_note_jump_speed()))
    lines.append("shuffle: {:.2f}".format(_map.get_shuffle()))
    lines.append("shufflePeriod: {:.2f}".format(_map.get_shuffle_period()))

    n_left_notes = len(_map.get_notes(NoteGroupType.LEFT))
    n_right_notes = len(_map.get_notes(NoteGroupType.RIGHT))

    n_normal_notes = n_left_notes + n_right_notes
    lines.append("total normal notes: {}".format(n_normal_notes))
    lines.append("total bombs: {}".format(len(_map.get_notes(NoteGroupType.BOMB))))

    lines.append("notes count (left,right): ({}, {})".format(n_left_notes, n_right_notes))

    left_note_fraction = n_left_notes / n_normal_notes
    right_note_fraction = n_right_notes / n_normal_notes

    lines.append("notes leaning (left-, right+): {:.2f}".format(_map.get_left_right_lean_fraction()))

    lines.append("average duration (s) between notes: {:.2f}".format(_map.get_average_duration_in_seconds_between_notes()))

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

def save_bar_charts_pdf(pdf_filepath, map_collections):
    with PdfPages(pdf_filepath) as pdf:
        build_bar_chart(map_collections, 'Number of songs', lambda c : c.get_number_of_maps())
        pdf.savefig()
        plt.close()
        build_bar_chart(map_collections, 'Duration (s)', lambda c : c.get_duration(), lambda c : c.get_duration_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(map_collections, 'Average Duration between Notes (s)', lambda c : c.get_average_duration_in_seconds_between_notes(), lambda c : c.get_average_duration_in_seconds_between_notes_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(map_collections, 'Left/Right(-/+) Lean', lambda c : c.get_left_right_lean_fraction(), lambda c : c.get_left_right_lean_fraction_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(map_collections, 'Note Jump Speed', lambda c : c.get_note_jump_speed(), lambda c : c.get_note_jump_speed_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(map_collections, 'BMP', lambda c : c.get_beats_per_minute(), lambda c : c.get_beats_per_minute_std())
        pdf.savefig()
        plt.close()
        build_bar_chart(map_collections, 'BMB', lambda c : c.get_beats_per_bar(), lambda c : c.get_beats_per_bar_std())
        pdf.savefig()
        plt.close()
        # build_bar_chart(map_collections, 'Shuffle', lambda c : c.get_shuffle(), lambda c : c.get_shuffle_std())
        # pdf.savefig()
        # plt.close()
        # build_bar_chart(map_collections, 'Shuffle period', lambda c : c.get_shuffle_period(), lambda c : c.get_shuffle_period_std())
        # pdf.savefig()
        # plt.close()

def build_bar_chart(map_collections, value_name, fn_val, fn_std=None, color='b'):
    n_difficulties = len(map_collections)

    values = tuple([fn_val(c) for c in map_collections])
    std_values = None
    if fn_std:
        std_values = tuple([fn_std(c) for c in map_collections])

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
    ax.set_xticklabels(tuple([c.difficulty for c in map_collections]))

    fig.tight_layout()

def save_pdf(pdf_filepath, _map):
    with PdfPages(pdf_filepath) as pdf:
        fig = plt.figure()
        plt.axis('off')
        plt.text(0.05,0.05, get_basic_data_as_text(_map), transform=fig.transFigure, size=14)
        pdf.savefig()
        plt.close()
        all_normal_notes = _map.get_notes(NoteGroupType.NORMAL)
        build_note_heatmap(all_normal_notes)
        plt.text(0, 2.75, "Total normal notes: {}".format(len(all_normal_notes)))
        pdf.savefig()
        plt.close()
        all_bomb_notes = _map.get_notes(NoteGroupType.BOMB)
        build_note_heatmap(all_bomb_notes, "Reds")
        plt.text(0, 2.75, "Total bombs: {}".format(len(all_bomb_notes)))
        pdf.savefig()
        plt.close()
        all_notes = _map.get_notes()
        build_histogram(all_notes)
        pdf.savefig()
        plt.close()
        build_cut_directions_drawing(all_normal_notes)
        pdf.savefig()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--path', help="Custom songs folder path.")
    parser.add_argument('--audio_filepath', help="Audio file path to use for the 'generate' operation.")
    parser.add_argument('--difficulty', type=str, choices=MAP_DIFFICULTIES, default=None, help='Difficulty of maps to analyze, if not set then analyze all difficulties')
    parser.add_argument('--text_filter', default=None, help='Text to use to filter maps.')
    parser.add_argument('--max_count', type=int, default=None, help="Numbers of maps to analyse, -1 then no maximum.")
    parser.add_argument('--output_filepath', type=str, default="OUTPUT", help='Filename to output to when performing certain operations.')
    parser.add_argument('--operations', nargs='+', choices=OPERATIONS, default=[OPERATIONS[0]], help='One or more operations to perform using the given songs/maps.')

    args = parser.parse_args()

    is_operation_handled = False
    if OPERATIONS[0] in args.operations:
        is_operation_handled = True
        map_collection = MapCollection(args.path, difficulty=args.difficulty, text_filter=args.text_filter, max_count=args.max_count)
        save_pdf("{}.pdf".format(args.output_filepath), map_collection)
        map_descriptors = ["{} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in map_collection.get_maps()]
        with open("{}.txt".format(args.output_filepath), 'wt', encoding='utf-8') as f:
            f.write('\n'.join(map_descriptors))
    if OPERATIONS[1] in args.operations:
        is_operation_handled = True
        map_collection = MapCollection(args.path, difficulty=args.difficulty, text_filter=args.text_filter, max_count=args.max_count)
        for s in map_collection.get_maps():
            save_pdf("{}_{}_{}_{}.pdf".format(args.output_filepath, s.ident, s.name, s.difficulty), s)
    if OPERATIONS[2] in args.operations:
        is_operation_handled = True
        map_collections = []
        lines = []
        for d in MAP_DIFFICULTIES:
            map_collection = MapCollection(args.path, difficulty=d, max_count=args.max_count, text_filter=args.text_filter)
            lines.append("{} - {} maps".format(d, map_collection.get_number_of_maps()))
            map_collections.append(map_collection)
            map_descriptors = ["  {} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in map_collection.get_maps()]
            lines.extend(map_descriptors)
        with open("{}.txt".format(args.output_filepath), 'wt', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        save_bar_charts_pdf("{}.pdf".format(args.output_filepath), map_collections)
    if not is_operation_handled and OPERATIONS[3] in args.operations:
        is_operation_handled = True

        map_collection = MapCollection(args.path, difficulty=args.difficulty, text_filter=args.text_filter, max_count=args.max_count)

        song = Song(args.audio_filepath)

        map_generators = [
            # ("Beat", 1, MapGeneratorBeatStrategy(song)),
            # ("Random", 3, MapGeneratorRandomStrategy(song)),
            # ("Weighted Random", 4, MapGeneratorWeightedRandomStrategy(song, map_collection)),
            ("Markov Chains", 5, MapGeneratorMarkovChainsStrategy(song, map_collection))
        ]
        song.generate_maps(args.output_filepath, map_generators)

    if not is_operation_handled:
        print("Unhandled operation type, error in code")
        sys.exit(1)

if __name__ == "__main__":
    main()
