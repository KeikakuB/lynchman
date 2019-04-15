import random
import collections

from .MapGeneratorStrategy import MapGeneratorStrategy
from api.data.Block import Block
from api.data.JsonNoteType import JsonNoteType
from api.data.NoteGroupType import NoteGroupType
import api.data.Constants

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

        # Count the number of notes with each cut direction based on the notes location on the grid
        n_cut_directions_by_grid_position = {}
        for (hand, index, layer) in blocks:
            counter = collections.Counter()
            for c in range(api.data.Constants.N_CUT_DIRECTIONS):
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
