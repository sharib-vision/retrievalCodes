#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:27:05 2018

@author: shariba

"""

'''
this paths.py is for autoencoding of the videos 
'''


import os

currentPath = os.path.abspath(os.path.dirname(__file__))

localCopy = 1

if localCopy:
    ROOT_DIR = currentPath
    DATA_DIR = os.path.join(ROOT_DIR, 'video')
    RESULT_DIR = os.path.join(ROOT_DIR, 'retrievalResults')
    PROJECT_DIR =  ROOT_DIR   

os.makedirs(RESULT_DIR, exist_ok=True)


print('VIDEO DATA_DIR: ', DATA_DIR)
print('RESULTS_DIR: ', RESULT_DIR)

