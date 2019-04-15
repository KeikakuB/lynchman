#!/usr/bin/env python

import sys
import api.plotting.Helper

import click

from api.data.Song import Song
from api.data.MapCollection import MapCollection

from api.generators.MapGeneratorBeatStrategy import MapGeneratorBeatStrategy
from api.generators.MapGeneratorRandomStrategy import MapGeneratorRandomStrategy
from api.generators.MapGeneratorWeightedRandomStrategy import MapGeneratorWeightedRandomStrategy
from api.generators.MapGeneratorMarkovChainsStrategy import MapGeneratorMarkovChainsStrategy

import api.data.Constants

@click.command()
@click.option('--path', help="Custom songs folder path.")
@click.option('--audio_filepath', help="Audio file path to use for the 'generate' operation.")
@click.option('--difficulty', type=click.Choice(api.data.Constants.MAP_DIFFICULTIES), default=None, help='Difficulty of maps to analyze, if not set then analyze all difficulties')
@click.option('--text_filter', default=None, help='Text to use to filter maps.')
@click.option('--max_count', type=int, default=None, help="Numbers of maps to analyse, -1 then no maximum.")
@click.option('--output_filepath', type=str, default="OUTPUT", help='Filename to output to when performing certain operations.')
@click.option('--operations', type=click.Choice(api.data.Constants.OPERATIONS), default=[api.data.Constants.OPERATIONS[0]], multiple=True, help='One or more operations to perform using the given songs/maps.')
def cli(path, audio_filepath, difficulty, text_filter, max_count, output_filepath, operations):
    is_operation_handled = False
    if api.data.Constants.OPERATIONS[0] in operations:
        is_operation_handled = True
        map_collection = MapCollection(path, difficulty=difficulty, text_filter=text_filter, max_count=max_count)
        api.plotting.Helper.save_pdf("{}.pdf".format(output_filepath), map_collection)
        map_descriptors = ["{} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in map_collection.get_maps()]
        with open("{}.txt".format(output_filepath), 'wt', encoding='utf-8') as f:
            f.write('\n'.join(map_descriptors))
    if api.data.Constants.OPERATIONS[1] in operations:
        is_operation_handled = True
        map_collection = MapCollection(path, difficulty=difficulty, text_filter=text_filter, max_count=max_count)
        for s in map_collection.get_maps():
            api.plotting.Helper.save_pdf("{}_{}_{}_{}.pdf".format(output_filepath, s.ident, s.name, s.difficulty), s)
    if api.data.Constants.OPERATIONS[2] in operations:
        is_operation_handled = True
        map_collections = []
        lines = []
        for d in api.data.Constants.MAP_DIFFICULTIES:
            map_collection = MapCollection(path, difficulty=d, max_count=max_count, text_filter=text_filter)
            lines.append("{} - {} maps".format(d, map_collection.get_number_of_maps()))
            map_collections.append(map_collection)
            map_descriptors = ["  {} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in map_collection.get_maps()]
            lines.extend(map_descriptors)
        with open("{}.txt".format(output_filepath), 'wt', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        api.plotting.Helper.save_bar_charts_pdf("{}.pdf".format(output_filepath), map_collections)
    if not is_operation_handled and api.data.Constants.OPERATIONS[3] in operations:
        is_operation_handled = True

        map_collection = MapCollection(path, difficulty=difficulty, text_filter=text_filter, max_count=max_count)

        song = Song(audio_filepath)

        map_generators = [
            ("Beat", 1, MapGeneratorBeatStrategy(song)),
            ("Random", 3, MapGeneratorRandomStrategy(song)),
            ("Weighted Random", 4, MapGeneratorWeightedRandomStrategy(song, map_collection)),
            ("Markov Chains DEBUG", 5, MapGeneratorMarkovChainsStrategy(song, map_collection, True)),
            ("Markov Chains", 5, MapGeneratorMarkovChainsStrategy(song, map_collection))
        ]
        song.generate_maps(output_filepath, map_generators)

    if not is_operation_handled:
        print("Unhandled operation type, error in code")
        sys.exit(1)
