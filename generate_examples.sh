CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"

rm ~/projects/lynchman/examples/*.pdf
rm ~/projects/lynchman/examples/*.txt
rm ~/projects/lynchman/examples/generated_maps/*.json

lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "examples/Koto" --text_filter Koto --difficulty Expert single
lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "examples/AggregatedSongsData" multi
lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "examples/DifficultiesCompared" compare
lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "examples/generated_maps/" generate --audio_filepath "maps/heart_on_wave.ogg"
