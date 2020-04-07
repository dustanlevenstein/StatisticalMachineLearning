#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:13:55 2020

@author: dustan
"""

import re
regex = re.compile("^.+(sub-.+)_(ses-.+)_(mod-.+)")
strings = ["abcsub-033_ses-01_mod-mri", "defsub-044_ses-01_mod-mri", "ghisub-055_ses-02_mod-ctscan" ]
print([regex.findall(s) for s in strings])

regex = re.compile("(sub-[^_]+)")
print([regex.sub("SUB-", s) for s in strings])

print(regex.sub("SUB-", "toto"))
print(re.sub('[^0-9a-zA-Z]+', '', 'h^&ell`.,|o w]{+orld'))