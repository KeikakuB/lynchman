CUSTOM_SONGS_PATH="/mnt/c/Program Files (x86)/Steam/steamapps/common/Beat Saber/CustomSongs/"
SCRIPT="python3 lynchman.py"
OUTPUT_PATH="~/projects/lynchman/examples/"

rm ~/projects/lynchman/examples/*.pdf
rm ~/projects/lynchman/examples/*.txt

${SCRIPT} --path "$CUSTOM_SONGS_PATH" --output_filepath "examples/Koto" --text_filter Koto --difficulty Expert --operations single
${SCRIPT} --path "$CUSTOM_SONGS_PATH" --output_filepath "examples/AggregatedSongsData" --operations multi
${SCRIPT} --path "$CUSTOM_SONGS_PATH" --output_filepath "examples/DifficultiesCompared" --operations compare
