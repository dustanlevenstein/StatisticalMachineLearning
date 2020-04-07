#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:30:05 2020

@author: dustan
"""

import os
WD = os.path.join(os.environ["HOME"], "git", "pystatsml", "datasets")
print(WD)
for dirpath, dirnames, filenames in os.walk(WD):
    print(dirpath, dirnames, filenames)
    
import glob
filenames = glob.glob(os.path.join(os.environ["HOME"], "Dropbox", "*",  "*.html"))
basenames = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
print(basenames)