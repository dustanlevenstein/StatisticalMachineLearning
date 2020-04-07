#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:59:30 2020

@author: dustan
"""

def calc(a, b, op):
    ops = {"multiply": lambda x, y: x*y,
           "add": lambda x, y: x+y,
           "divide": lambda x, y: x/y,
           "subtract": lambda x, y: x-y,
           "exponentiate": lambda x, y: x**y}
    if op in ops:
        return ops[op](a, b)
    else:
        raise ValueError("Invalid operation")