from enum import Enum

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
