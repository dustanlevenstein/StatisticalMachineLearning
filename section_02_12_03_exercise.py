#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:10:12 2020

@author: dustan
"""

import re

def parse_word_count(filename = "section_02_12_03_bsd4clause.txt"):
    regex = re.compile("[a-zA-Z]+")
    f = open(filename, "r")
    count = dict()
    for line in f:
        for word in regex.findall(line.lower()):
            count[word] = count.get(word, 0) + 1
    f.close()
    return count

import argparse
import os
import os.path
import pandas as pd
def to_file():
    # parse command line options
    output = "word_count.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='list of input files.',
                        nargs='+', type=str)
    parser.add_argument('-o', '--output',
                        help='output csv file (default %s)' % output,
                        type=str, default=output)
    options = parser.parse_args()
    if options.input is None :
        parser.print_help()
        raise SystemExit("Error: input files are missing")
    else:
        filenames = [f for f in options.input if os.path.isfile(f)]
    # Match words
    regex = re.compile("[a-zA-Z]+")
    count = dict()
    for filename in filenames:
        fd = open(filename, "r")
        for line in fd:
            for word in regex.findall(line.lower()):
                count[word] = count.get(word, 0) + 1
    fd = open(options.output, "w")
    # Pandas
    df = pd.DataFrame([[k, count[k]] for k in count], columns=["word", "count"])
    df.to_csv(options.output, index=False)
    
if __name__ == "__main__":
    to_file()