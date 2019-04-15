import json

from mutagen.oggvorbis import OggVorbis

from .NoteGroupType import NoteGroupType

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

