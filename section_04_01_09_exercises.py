#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:37:29 2020

@author: dustan
"""

import pandas as pd


def simple_linear_regression_and_correlation():
    df = pd.read_csv("ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/datasets/birthwt.csv")
    print(df.head())