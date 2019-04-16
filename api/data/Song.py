import os
import json

os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
os.environ['LIBROSA_CACHE_LEVEL'] = '30'
import librosa

import api.data.Constants

class Song:
    def __init__(self, audio_filepath):
        self._audio_filepath = audio_filepath

        # Load the audio as a waveform `y` and the sampling rate as `sr`
        self._y, self._sr = librosa.load(audio_filepath)

        # Run the default beat tracker
        self._tempo, self._beat_frames = librosa.beat.beat_track(y=self._y, sr=self._sr)

        # Convert the frame indices of beat events into timestamps
        self._beat_times = librosa.frames_to_time(self._beat_frames, sr=self._sr)

        self._beats_per_minute = round(float("{:.2f}".format(self._tempo)))

    def get_beat_times(self):
        return self._beat_times

    def get_beats_per_minute(self):
        return self._beats_per_minute

    def _generate_map(self, output_directory, difficulty_name, map_generator):
        map_generator.generate()

        in_events = map_generator.get_events()
        in_notes = map_generator.get_notes()
        in_obstacles = map_generator.get_obstacles()

        data = {}
        data["_version"] = "1.5.0"
        data["_beatsPerMinute"] = self.get_beats_per_minute()
        data["_beatsPerBar"] = 16
        data["_noteJumpSpeed"] = 10
        data["_shuffle"] = 0
        data["_shufflePeriod"] = 0.5
        data["_time"] = 0
        data["_events"] = in_events
        data["_notes"] = in_notes
        data["_obstacles"] = in_obstacles
        data["_bookmarks"] = []

        with open(os.path.join(output_directory, "{}.json".format(difficulty_name)), 'w') as outfile:
            json.dump(data, outfile)

    def generate_maps(self, output_directory, map_generators):
        for (name, _, generator) in map_generators:
            self._generate_map(output_directory, name, generator)

        audio_filename = os.path.basename(self._audio_filepath)
        (audio_name, _) = os.path.splitext(audio_filename)
        info_data = {}
        info_data["authorName"] = "KeikakuB"
        info_data["beatsPerMinute"] = self.get_beats_per_minute()
        info_data["coverImagePath"] = "cover.jpg"
        difficulty_levels = []
        for (name, rank, generator) in map_generators:
            difficulty_levels.append({
                "audioPath": audio_filename,
                "difficulty": "Expert",
                "difficultyLabel": name,
                "difficultyRank": rank,
                "jsonPath": "{}{}".format(name, api.data.Constants.MAP_EXTENSION),
                "offset": 0,
                "oldOffset": 0
            }
            )
        info_data["difficultyLevels"] = difficulty_levels
        info_data["environmentName"] =  "DefaultEnvironment"
        info_data["previewDuration"] =  10
        info_data["previewStartTime"] =  12
        info_data["songName"] =  audio_name
        info_data["songSubName"] =  ""

        with open(os.path.join(output_directory, "info.json"), 'w') as outfile:
            json.dump(info_data, outfile)
