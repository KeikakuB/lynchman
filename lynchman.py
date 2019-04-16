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

ARG_CUSTOM_SONGS_DIRECTORY = 'CUSTOM_SONGS_DIRECTORY'
ARG_OUTPUT_PATH_PREFIX = 'OUTPUT_PATH_PREFIX'
ARG_DIFFICULTY = 'DIFFICULTY'
ARG_TEXT_FILTER = 'TEXT_FILTER'
ARG_MAX_COUNT = 'MAX_COUNT'

@click.group()
@click.option('--songs_dir', required=True, help="`CustomSongs` directory path.")
@click.option('--output_path_prefix', required=True, help='Path prefix for output files.')
@click.option('--difficulty', type=click.Choice(api.data.Constants.MAP_DIFFICULTIES), default=None, help='Difficulty of maps to analyze, if not set then analyze all difficulties.')
@click.option('--text_filter', default=None, help='Text to use to filter maps.')
@click.option('--max_count', type=int, default=None, help="Numbers of maps to analyse, -1 then no maximum.")
@click.pass_context
def cli(ctx, songs_dir, output_path_prefix, difficulty, text_filter, max_count):
    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj[ARG_CUSTOM_SONGS_DIRECTORY] = songs_dir
    ctx.obj[ARG_DIFFICULTY] = difficulty
    ctx.obj[ARG_TEXT_FILTER] = text_filter
    ctx.obj[ARG_MAX_COUNT] = max_count
    ctx.obj[ARG_OUTPUT_PATH_PREFIX] = output_path_prefix

@cli.command(help='Generate a PDF analyzing a group of maps.')
@click.pass_context
def multi(ctx):
    map_collection = MapCollection(ctx.obj[ARG_CUSTOM_SONGS_DIRECTORY], difficulty=ctx.obj[ARG_DIFFICULTY], text_filter=ctx.obj[ARG_TEXT_FILTER], max_count=ctx.obj[ARG_MAX_COUNT])
    api.plotting.Helper.save_pdf("{}.pdf".format(ctx.obj[ARG_OUTPUT_PATH_PREFIX]), map_collection)
    map_descriptors = ["{} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in map_collection.get_maps()]
    with open("{}.txt".format(ctx.obj[ARG_OUTPUT_PATH_PREFIX]), 'wt', encoding='utf-8') as f:
        f.write('\n'.join(map_descriptors))

@cli.command(help='Generate a PDF on each map.')
@click.pass_context
def single(ctx):
    map_collection = MapCollection(ctx.obj[ARG_CUSTOM_SONGS_DIRECTORY], difficulty=ctx.obj[ARG_DIFFICULTY], text_filter=ctx.obj[ARG_TEXT_FILTER], max_count=ctx.obj[ARG_MAX_COUNT])
    for s in map_collection.get_maps():
        api.plotting.Helper.save_pdf("{}_{}_{}_{}.pdf".format(ctx.obj[ARG_OUTPUT_PATH_PREFIX], s.ident, s.name, s.difficulty), s)

@cli.command(help='Generate a PDF comparing maps based on their difficulty.')
@click.pass_context
def compare(ctx):
    map_collections = []
    lines = []
    for d in api.data.Constants.MAP_DIFFICULTIES:
        map_collection = MapCollection(ctx.obj[ARG_CUSTOM_SONGS_DIRECTORY], difficulty=d, text_filter=ctx.obj[ARG_TEXT_FILTER], max_count=ctx.obj[ARG_MAX_COUNT])
        lines.append("{} - {} maps".format(d, map_collection.get_number_of_maps()))
        map_collections.append(map_collection)
        map_descriptors = ["  {} _ {} _ {}".format(s.ident, s.name, s.difficulty) for s in map_collection.get_maps()]
        lines.extend(map_descriptors)
    with open("{}.txt".format(ctx.obj[ARG_OUTPUT_PATH_PREFIX]), 'wt', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    api.plotting.Helper.save_bar_charts_pdf("{}.pdf".format(ctx.obj[ARG_OUTPUT_PATH_PREFIX]), map_collections)

@cli.command(help='Generate a map from a given audio file.')
@click.option('--audio_filepath', help="Audio file path.")
@click.pass_context
def generate(ctx, audio_filepath):
    map_collection = MapCollection(ctx.obj[ARG_CUSTOM_SONGS_DIRECTORY], difficulty=ctx.obj[ARG_DIFFICULTY], text_filter=ctx.obj[ARG_TEXT_FILTER], max_count=ctx.obj[ARG_MAX_COUNT])

    song = Song(audio_filepath)

    map_generators = [
        ("Beat", 1, MapGeneratorBeatStrategy(song)),
        ("Random", 3, MapGeneratorRandomStrategy(song)),
        ("Weighted Random", 4, MapGeneratorWeightedRandomStrategy(song, map_collection)),
        ("Markov Chains DEBUG", 5, MapGeneratorMarkovChainsStrategy(song, map_collection, True)),
        ("Markov Chains", 5, MapGeneratorMarkovChainsStrategy(song, map_collection))
    ]
    song.generate_maps(ctx.obj[ARG_OUTPUT_PATH_PREFIX], map_generators)
