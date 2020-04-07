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
