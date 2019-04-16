CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"

# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --difficulty Easy --output_path_prefix "out/All_Easy" multi
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --difficulty Normal --output_path_prefix "out/All_Normal" multi
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --difficulty Hard --output_path_prefix "out/All_Hard" multi
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --difficulty Expert --output_path_prefix "out/All_Expert" multi
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --difficulty ExpertPlus --output_path_prefix "out/All_ExpertPlus" multi
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "out/All_All" multi
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "out/KOTO" --text_filter Koto --difficulty Expert single
# lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "out/Compare" compare
lynchman --songs_dir "$CUSTOM_SONGS_PATH" --output_path_prefix "/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/WIP Songs/very_spicey_meme/" generate --audio_filepath "maps/heart_on_wave.ogg"
