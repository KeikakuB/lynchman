from collections import namedtuple

Block = namedtuple('Block', ['type', 'coords', 'cut_direction'])

def make_block_from_note(n):
    return Block(n["_type"], (n["_lineIndex"], n["_lineLayer"]), n["_cutDirection"])
