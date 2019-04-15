from enum import Enum

class NoteGroupType(Enum):
    """Enum used for picking which type of notes we're looking at."""
    NONE = 1
    ALL = 2
    NORMAL = 3
    LEFT = 4
    RIGHT = 5
    BOMB = 6

