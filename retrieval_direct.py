#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:34:26 2019

@author: shariba
"""


from retrieveQuerySequences import retrieve_closest_images
from imageCroppingClasses import imageProperties
import time
import os
import numpy as np

from paths import DATA_DIR, ROOT_DIR,  RESULT_DIR, PROJECT_DIR


import argparse
parser_vid = argparse.ArgumentParser()
parser_vid.add_argument('-videoFile', type=str, default='M_01062017090724_0000000000001772_1_001_001-1', help='enter checkpoint directory')
parser_vid.add_argument('-useTestFlag', type=int, default=0, help='enter checkpoint directory')
args_vid = parser_vid.parse_args()

#TODO parse it
gpu = 0

if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args_vid.gpu_id)
    
if args_vid.useTestFlag:
    args_vid.videoFile='M_01032018130721_0000000000003059_1_002_001-1'
    args_vid.videoFile= 'M_01032017120001_0000000000001603_1_001_001-1'
    args_vid.videoFile= 'M_10082017122840_0000000000002470_1_001_001-1'
    args_vid.videoFile= 'M_01062017090724_0000000000001772_1_001_001-1'

args_vid.videoFile=args_vid.videoFile+'.MP4'
videoFileName = args_vid.videoFile
outfile=videoFileName.split('.')[0]   

frame_scores = np.load(RESULT_DIR+'/'+ outfile+'_frameScore'+'.npy')
print('imageLists of the embedding loaded... continuing to query image file')


# retrive list
frame_scoreOriginal = np.load(RESULT_DIR+'/'+ outfile+'_frameScoreOriginal'+'.npy')

print(frame_scoreOriginal)

imageDirPath = os.path.join(PROJECT_DIR, outfile)

textFileName=outfile+'_retrievedImages.txt'
textfile = open(textFileName, 'a')
frameWithOnes=[i for i,x in enumerate(frame_scores) if x == 1]

for i in range (0, len(frame_scoreOriginal)):
    for j in range (0, len(frame_scoreOriginal[i])):
        textfile.write(os.path.join(imageDirPath, str(frameWithOnes[frame_scoreOriginal[i][j]]))+'.jpg')

        textfile.write("\n")

textfile.close()