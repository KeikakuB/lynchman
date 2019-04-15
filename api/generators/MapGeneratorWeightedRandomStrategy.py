import random
import collections

import api.data.Constants
from .MapGeneratorStrategy import MapGeneratorStrategy
from api.data.Block import Block
from api.data.JsonNoteType import JsonNoteType
from api.data.NoteGroupType import NoteGroupType
from api.utils.ProbabilityCounter import ProbabilityCounter

class MapGeneratorWeightedRandomStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which notes are placed randomly on every beat
        with the probabilities of each note being determined by its representation in
        the given `map_collection`."""

    def __init__(self, song, map_collection):
        MapGeneratorStrategy.__init__(self, song)
        self._map_collection = map_collection

    def _generate(self):
        blocks = []
        for hand in range(api.data.Constants.N_HANDS):
            for index in range(api.data.Constants.N_LINE_INDEX):
                for layer in range(api.data.Constants.N_LINE_LAYER):
                    blocks.append((hand, index, layer))

        cut_directions_by_grid_position_counters = {}
        notes_counter = collections.Counter()
        for m in self._map_collection.get_maps():
            for n in m.get_notes(NoteGroupType.NORMAL):
                block = (n["_type"], n["_lineIndex"], n["_lineLayer"])
                cut_direction = n["_cutDirection"]
                if block not in cut_directions_by_grid_position_counters:
                    cut_directions_by_grid_position_counters[block] = collections.Counter()
                cut_directions_by_grid_position_counters[block][cut_direction] += 1
                notes_counter[block] += 1

        cut_directions_by_grid_position_probability_counters = {}
        for block in blocks:
            cut_directions_by_grid_position_probability_counters[block] = ProbabilityCounter(cut_directions_by_grid_position_counters[block])

        notes_probability_counter = ProbabilityCounter(notes_counter)
        beat_times = self._song.get_beat_times()
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            (hand, line_index, line_layer) = notes_probability_counter.get()
            note_type = JsonNoteType.RIGHT
            if hand == JsonNoteType.LEFT.value:
                note_type = JsonNoteType.LEFT
            cut_direction = cut_directions_by_grid_position_probability_counters[(hand, line_index, line_layer)].get()
            self._add_note(float("{:.16f}".format(b)), Block(note_type.value, (line_index, line_layer), cut_direction))
