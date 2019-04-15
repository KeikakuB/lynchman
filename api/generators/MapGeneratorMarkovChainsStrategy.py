import random
import collections
import math
from operator import itemgetter, attrgetter

from .MapGeneratorStrategy import MapGeneratorStrategy
import api.data.Block
from api.data.Block import Block
from api.data.NoteGroupType import NoteGroupType
from api.utils.ProbabilityCounter import ProbabilityCounter

class MapGeneratorMarkovChainsStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which notes are placed randomly on every beat
        based on Markov chains determined the given `map_collection`."""

    def __init__(self, song, map_collection, debug=False):
        MapGeneratorStrategy.__init__(self, song)
        self._map_collection = map_collection
        self._debug = debug

    def _generate(self):
        patterns_by_map = []
        time_pattern_tuples_by_map = []
        for m in self._map_collection.get_maps():
            notes = m.get_notes(NoteGroupType.NORMAL)
            # Group notes into 'patterns' if they occur at the same time
            patterns = []
            time_pattern_tuples = []
            current_pattern = [api.data.Block.make_block_from_note(notes[0])]
            for i in range(len(notes) - 1):
                l = notes[i]
                r = notes[i + 1]
                # TODO play with the fuzzy value -> but 1e-8 seems to work well
                if math.isclose(l["_time"], r["_time"], rel_tol=1e-8):
                    # Append right block and continue
                    current_pattern.append(api.data.Block.make_block_from_note(r))
                else:
                    # Sort `current_pattern` to ensure uniqueness
                    current_pattern = tuple(sorted(current_pattern, key=attrgetter('type', 'coords', 'cut_direction')))
                    time_pattern_tuples.append((l["_time"], current_pattern))
                    patterns.append(current_pattern)
                    current_pattern = [api.data.Block.make_block_from_note(r)]
            time_pattern_tuples_by_map.append(time_pattern_tuples)
            patterns_by_map.append(patterns)

        patterns_count = collections.Counter([item for sublist in patterns_by_map for item in sublist])
        patterns_probability_counter = ProbabilityCounter(patterns_count)

        pattern_adjacency_counts = {}
        for c in patterns_probability_counter.keys():
            pattern_adjacency_counts[c] = collections.Counter()

        allowed_delay_in_seconds = 2 * (60 / self._song.get_beats_per_minute())
        for patterns in time_pattern_tuples_by_map:
            for i in range(len(patterns) - 1):
                (t1, l) = patterns[i]
                (t2, r) = patterns[i + 1]
                if math.isclose(t1, t2, rel_tol=allowed_delay_in_seconds):
                    pattern_adjacency_counts[l][r] += 1

        patterns_adjacency_probability_counters = {}
        for (pattern, count) in pattern_adjacency_counts.items():
            patterns_adjacency_probability_counters[pattern] = ProbabilityCounter(pattern_adjacency_counts[pattern])

        beat_times = self._song.get_beat_times()
        if self._debug:
            patterns_ordered_by_probability = patterns_probability_counter.keys()
            pattern_index = 0
            for i in range(0, len(beat_times), 2):
                b = beat_times[i]
                for block in patterns_ordered_by_probability[pattern_index]:
                    self._add_note(float("{:.16f}".format(b)), block)
                pattern_index += 1
            return
        # todo figure out why notes stop early (check beat_times)
        current_pattern = patterns_probability_counter.get()
        n_patterns_in_current_sequence = 0
        n_sequence_pattern_limit = 31
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            if not current_pattern:
                # print("Resetting pattern sequence after {}".format(b))
                current_pattern =  patterns_probability_counter.get()
                # skip a beat to allow the player to reposition if needed
                continue
            for block in current_pattern:
                self._add_note(float("{:.16f}".format(b)), block)
            n_patterns_in_current_sequence += 1
            if n_patterns_in_current_sequence >= n_sequence_pattern_limit:
                current_pattern = None
                n_patterns_in_current_sequence = 0
            else:
                current_pattern = patterns_adjacency_probability_counters[current_pattern].get()
