import random

from .MapGeneratorStrategy import MapGeneratorStrategy
from api.data.Block import Block
from api.data.JsonNoteType import JsonNoteType
import api.data.Constants

class MapGeneratorRandomStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which notes are placed completely randomly on every beat."""
    def __init__(self, song):
        MapGeneratorStrategy.__init__(self, song)

    def _generate(self):
        beat_times = self._song.get_beat_times()
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]
            hand = random.randrange(api.data.Constants.N_HANDS)
            note_type = JsonNoteType.RIGHT
            if hand == JsonNoteType.LEFT.value:
                note_type = JsonNoteType.LEFT
            line_index = random.randrange(api.data.Constants.N_LINE_INDEX)
            line_layer = random.randrange(api.data.Constants.N_LINE_LAYER)
            cut_direction = random.randrange(api.data.Constants.N_CUT_DIRECTIONS)
            self._add_note(float("{:.16f}".format(b)), Block(note_type.value, (line_index, line_layer), cut_direction))
