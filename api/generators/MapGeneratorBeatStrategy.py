from .MapGeneratorStrategy import MapGeneratorStrategy
from api.data.Block import Block
import api.data.Constants
from api.data.JsonNoteType import JsonNoteType

class MapGeneratorBeatStrategy(MapGeneratorStrategy):
    """This strategy generates a map in which the same note is placed on every beat."""
    def __init__(self, song):
        MapGeneratorStrategy.__init__(self, song)

    def _generate(self):
        beat_times = self._song.get_beat_times()
        for i in range(0, len(beat_times), 2):
            b = beat_times[i]

            self._add_note(float("{:.16f}".format(b)), Block(JsonNoteType.RIGHT.value, (2, 0) , api.data.Constants.N_CUT_DIRECTIONS - 1))
