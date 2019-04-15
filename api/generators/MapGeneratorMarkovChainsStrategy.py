import random
import collections
import math
from operator import itemgetter, attrgetter

from .MapGeneratorStrategy import MapGeneratorStrategy
import api.data.Block
from api.data.Block import Block
from api.data.NoteGroupType import NoteGroupType

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
                # print("Resetting pattern sequence after {}".format(b))
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
