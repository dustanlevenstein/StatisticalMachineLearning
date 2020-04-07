#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:06:04 2020

@author: dustan
"""

def delete_adj_duplicates(li):
    previous = None
    result = []
    for ii in li:
        if ii != previous:
            result.append(ii)
        previous = ii
    return result
def delete_all_duplicates(li):
    previous = set()
    result = []
    for ii in li:
        if ii not in previous:
            result.append(ii)
            previous.add(ii)
    return result
