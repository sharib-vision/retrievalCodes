#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:18:49 2019

@author: shariba
"""
import argparse
from shutil import copyfile
import os
import errno

from paths import RESULT_DIR


def get_args():
    parser = argparse.ArgumentParser(description='endoscopic image retrieval')
    parser.add_argument('--txtFile', type=str, default='M_01062017090724_0000000000001772_1_001_001-1_retrievedImages.txt', help='training datalist')
    parser.add_argument('--retrivalDir', type=str, default='retrievedImageFromList', help='training datalist')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    
    args = get_args()
    txtFileTORead = args.txtFile
    print('txtfile',txtFileTORead)
    
    vidIdentity=txtFileTORead.split('_')[2].strip("0")
    
    dataList = open(txtFileTORead, 'rt').read().split('\n')
    
    RESULT_ = os.path.join(RESULT_DIR, args.retrivalDir, vidIdentity)
    try:
        os.makedirs(RESULT_)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(RESULT_):
            pass
        
    
    for i in range(0, len(dataList)-1):
        fileName = dataList[i]
        copyfile(fileName, os.path.join(RESULT_, fileName.split('/')[-1]))
        print(fileName.split('/')[-1])
#        print(i)