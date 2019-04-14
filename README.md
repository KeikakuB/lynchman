# Goal
The goal of the project is to develop code for computing data based on Beat Saber (a Virtual Reality rythm game whereby the player is tasked with using their arms to slice fast moving blocks in time with music) maps which can be used for interesting purposes such as generating maps automatically, recommending other songs based on a given player's favorite maps/songs, etc.

The name of the repo is based on the character Lynchman from one of the best animated movies of all time, Redline. He has a minor role in the movie but as an intergalactic mercenary for hire he spends a portion of his screen time stretching his arms and legs in preparation for his work, as one usually or should do before playing Beat Saber.

#  Reminders
- Conversion from .mp3 to .ogg must be done through audacity currently, a linux based solution would be nice (ffmpeg with the correct settings)

# JSON File Format for Beat Saber Maps
Maps are stored in .json files with a pretty self-explanatory format for the most part. The filename is the difficulty of the map. A song can have multiple maps and thus difficulties associated to it.

## Notes
### lineIndex (columns), lineLayer (rows)
Denotes the position of the block in a fixed 3x4 grid.

```
           --- --- --- ---
l       2 |   |   |   |   |
 a        |   |   |   |   |
  y        --- --- --- ---
   e    1 |   |   |   |   |
    r     |   |   |   |   |
           --- --- --- ---
        0 |   |   |   |   |
          |   |   |   |   |
           --- --- --- ---
            0   1   2   3
                index

```
lineLayer denotes the row as described in the diagram above.

lineIndex denotes the column as described in the diagram above.

### type
Denotes the type of block.

When type is 0, the block must be hit with the left saber.
When type is 1, the block must be hit with the right saber.
When type is 2?
When type is 3, the block is a bomb block.

### cutDirection
Denotes whether the block (which are boxes geometrically) is placed straight or on a diagonal as well as the cardinal direction, or lack thereof, through which the block must be cut.

```
     1
   -----
  |     |
3 |  8  | 2
  |     |
   -----
     0

     -
    / \
 7 /   \ 6
  /     \
  \     /
 5 \   / 4
    \ /
     -
```

NB: A value of 8 signifies that the block will be placed straight and can be cut from any direction.

## Events
???


