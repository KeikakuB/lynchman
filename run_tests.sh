CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"

# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Easy --output_filepath "out/All_Easy" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Normal --output_filepath "out/All_Normal" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Hard --output_filepath "out/All_Hard" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Expert --output_filepath "out/All_Expert" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty ExpertPlus --output_filepath "out/All_ExpertPlus" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --output_filepath "out/All_All" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --text_filter Koto --difficulty Expert  --op single
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --output_filepath "out/Compare" --op compare
python3 lynchman.py --audio_filepath "maps/heart_on_wave.ogg" --output_filepath "/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/very_spicey_meme/" --op generate
