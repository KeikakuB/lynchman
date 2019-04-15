import os

import numpy as np

from .NoteGroupType import NoteGroupType
from .Map import Map
import api.data.Constants

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
                            if os.path.isfile(filepath) and ext == api.data.Constants.SONG_EXTENSION:
                                audio_filepath = filepath
                            if os.path.isfile(filepath) and ext == api.data.Constants.MAP_EXTENSION:
                                if self.difficulty:
                                    is_map = filename == self.difficulty
                                else:
                                    is_map = filename in api.data.Constants.MAP_DIFFICULTIES
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

