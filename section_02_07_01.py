#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:24:56 2020

@author: dustan
"""

import tempfile
tmpdir = tempfile.gettempdir()
print(tmpdir)

import os
mytmpdir = os.path.join(tmpdir, "foobar")
print(os.listdir(tmpdir))
os.makedirs(os.path.join(tmpdir, "foobar", "plop", "toto"))
print(tmpdir)
