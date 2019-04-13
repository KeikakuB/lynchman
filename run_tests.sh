CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"

# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Easy --output_filename "All_Easy" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Normal --output_filename "All_Normal" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Hard --output_filename "All_Hard" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty Expert --output_filename "All_Expert" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --difficulty ExpertPlus --output_filename "All_ExpertPlus" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --output_filename "All_All" --op multi
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --text_filter Koto --difficulty Expert  --op single
python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --output_filename "Compare" --op compare
