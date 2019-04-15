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
