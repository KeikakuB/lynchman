CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"

# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op cumulative --difficulty Easy --output_filename "All_Easy"
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op cumulative --difficulty Normal --output_filename "All_Normal"
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op cumulative --difficulty Hard --output_filename "All_Hard"
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op cumulative --difficulty Expert --output_filename "All_Expert"
# python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op cumulative --difficulty ExpertPlus --output_filename "All_ExpertPlus"
python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op cumulative --output_filename "All_All"
python3 lynchman.py --path "$CUSTOM_SONGS_PATH" --op single --filter Koto --difficulty Expert 
