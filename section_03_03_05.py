#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:35:17 2020

@author: dustan
"""

# Time series
import seaborn as sns
sns.set(style="darkgrid")
# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
# Plot the responses for different events and regions
ax = sns.pointplot(x="timepoint", y="signal",
                   hue="region", style="event",
                   data=fmri)
# version 0.9
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)



# =============================================================================
#     In [21]: fmri.head()
#     Out[21]: 
#       subject  timepoint event    region    signal
#     0     s13         18  stim  parietal -0.017552
#     1      s5         14  stim  parietal -0.080883
#     2     s12         18  stim  parietal -0.081033
#     3     s11         18  stim  parietal -0.046134
#     4     s10         18  stim  parietal -0.037970
# =============================================================================
