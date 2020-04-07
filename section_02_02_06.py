#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:05:41 2020

@author: dustan
"""

# create an empty set
empty_set = set()
# create a set
languages = {'python', 'r', 'java'} # create a set directly
snakes = set(['cobra', 'viper', 'python']) # create a set from a list
# examine a set
len(languages)
'python' in languages # returns 3
# returns True
# set operations
languages & snakes
languages | snakes
languages - snakes
snakes - languages

languages.add('sql')
