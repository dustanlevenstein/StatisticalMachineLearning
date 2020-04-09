#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:02:12 2020

@author: dustan
"""

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 50)
sinus = np.sin(x)


### bitmap format
plt.plot(x, sinus)
plt.savefig("sinus.png")
plt.close()
# Prefer vectorial format (SVG: Scalable Vector Graphics) can be edited with
# Inkscape, Adobe Illustrator, Blender, etc.
plt.plot(x, sinus)
plt.savefig("sinus.svg")
plt.close()
# Or pdf
plt.plot(x, sinus)
plt.savefig("sinus.pdf")
plt.close()