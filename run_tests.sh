CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"

# lynchman --path "$CUSTOM_SONGS_PATH" --difficulty Easy --output_filepath "out/All_Easy" --operations multi
# lynchman --path "$CUSTOM_SONGS_PATH" --difficulty Normal --output_filepath "out/All_Normal" --operations multi
# lynchman --path "$CUSTOM_SONGS_PATH" --difficulty Hard --output_filepath "out/All_Hard" --operations multi
# lynchman --path "$CUSTOM_SONGS_PATH" --difficulty Expert --output_filepath "out/All_Expert" --operations multi
# lynchman --path "$CUSTOM_SONGS_PATH" --difficulty ExpertPlus --output_filepath "out/All_ExpertPlus" --operations multi
# lynchman --path "$CUSTOM_SONGS_PATH" --output_filepath "out/All_All" --operations multi
# lynchman --path "$CUSTOM_SONGS_PATH" --output_filepath "out/KOTO" --text_filter Koto --difficulty Expert  --operations single
# lynchman --path "$CUSTOM_SONGS_PATH" --output_filepath "out/Compare" --operations compare
lynchman --path "$CUSTOM_SONGS_PATH" --audio_filepath "maps/heart_on_wave.ogg" --output_filepath "/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/WIP Songs/very_spicey_meme/" --operations generate
