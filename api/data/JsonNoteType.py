from enum import Enum

class JsonNoteType(Enum):
    """JSON values for the `_type` field of a note."""
    LEFT = 0
    RIGHT = 1
    BOMB = 3
