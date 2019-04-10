#!/usr/bin/env python

import numpy as np
import pandas as pd
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("python3 banalyze.py JSON_FILEPATH")
        sys.exit(1)
    input_filename = sys.argv[1]
    with open(input_filename) as f:
        data = json.load(f)
    # {"_version":"1.5.0","_beatsPerMinute":100,"_beatsPerBar":16,"_noteJumpSpeed":10,"_shuffle":0,"_shufflePeriod":0.5,"_events":[{"_time":6.7333335876464844,"_type":13
    header_data_names = ["_version", "_beatsPerMinute", "_beatsPerBar", "_noteJumpSpeed", "_shuffle", "_shufflePeriod"]
    print("HEADERS")
    for n in header_data_names:
        print("{}: {}".format(n[1:], data[n]))

    print("EVENTS")
    events = data["_events"]
    for e in events:
        print("{}".format(e))

    print("NOTES")
    notes = data["_notes"]
    for n in notes:
        print("{}".format(n))

    #_time, _lineIndex, _lineLayer, _type, _cutDirection











    # TODO input json into numpy for manipulation
    # TODO compute stuff?


if __name__ == "__main__":
    main()
