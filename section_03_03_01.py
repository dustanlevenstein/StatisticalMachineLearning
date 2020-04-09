#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:03:48 2020

@author: dustan
"""

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 50)
sinus = np.sin(x)
cosinus = np.cos(x)
plt.plot(x, sinus, "-b", x, sinus, "ob", x, cosinus, "-r", x, cosinus, "or")
plt.xlabel("this is x factorial")
plt.ylabel("this is y!")
plt.title("Hello woild!")
plt.show()

plt.plot(x, sinus, label="sinus", color='blue', linestyle='--', linewidth=2) # I notice there's no documentation on plot.
plt.plot(x, cosinus, label="cosinus", color='red', linestyle='-', linewidth=2)
plt.legend()
plt.show()
