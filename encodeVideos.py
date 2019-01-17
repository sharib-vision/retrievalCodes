#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:04:00 2018

@author: shariba
"""

'''
    - extract video frames
    - list all the videos that you want to encode
    - encode using multi-threading for all the videos that has been put as array
    - only embeddings which doesnot exist are done here!!!
    

'''

import numpy as np
import time
import os
from imageCroppingClasses import imageDebugging
from paths import DATA_DIR, ROOT_DIR, RESULT_DIR, PROJECT_DIR


import argparse

'''
parse here:
    1) video file
    2) gpu id
    3) useTest or no

'''
parser_vid = argparse.ArgumentParser()
parser_vid.add_argument('-videoFile', type=str, default='M_01032018130721_0000000000003059_1_002_001-1', help='enter checkpoint directory')
parser_vid.add_argument('-gpu_id', type=int, default=3, help='enter gpu number')
parser_vid.add_argument('-useTestFlag', type=int, default=1, help='enter test case or not!')
parser_vid.add_argument('-checkpointDir', type=str, default='checkpoints', help='enter checkpoint directory')
args_vid = parser_vid.parse_args()



args_vid.videoFile=args_vid.videoFile+'.MP4'

print("arguments~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(args_vid.videoFile)
print(args_vid.gpu_id)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

gpu = 0

useTest = args_vid.useTestFlag

if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args_vid.gpu_id)
    #os.environ["CUDA_VISIBLE_DEVICES"]=args_vid.gpu_id
    
if useTest:
    vFiles = ['M_11052017100716_0000000000001716_1_001_001-1_1.MP4', 
              'M_11012018124919_0000000000002470_1_001_001-1_1.MP4', 
              'M_01032018130721_0000000000003059_1_002_001-1.MP4' ]
else:
    vFiles = ['M_02112017102541_0000000000001445_1_001_001-1.MP4', 
              'M_04052017121125_0000000000002468_1_001_001-1.MP4', 
              'M_06072017102000_0000000000002044_1_001_001-1.MP4',
              'M_08032018121821_0000000000003080_1_001_001-1.MP4',
              'M_15032018115410_0000000000002043_1_001_001-1.MP4']

videoFileName = args_vid.videoFile

if useTest:
    videoFileName = vFiles[2]
    videoFile = os.path.join(ROOT_DIR, 'dysplasiaEndoscopy', 'videos', videoFileName)
else:
    videoFile = os.path.join(DATA_DIR, videoFileName)
    
#patientFileName=videoFile.split('_')[1]+'_'+videoFile.split('_')[2].strip("0")
    
outfile=videoFileName.split('.')[0] 
patientFileName=outfile

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Video cleaning using our binary classification with DNN
    input - raw video
    output - clean video
    save as - cleanVideo/vFileName
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
outfile=videoFileName.split('.')[0]

#os.makedirs((os.path.join(RESULT_DIR, 'dysplasiaEndoscopy', 'cleanVideos')), exist_ok=True)
exists = os.path.isfile(RESULT_DIR+'/'+ outfile+'_frameScore'+'.npy')
print('videoFile cleaned flag:', exists)


debug = 0
runPart1 = 1

useCleanVideo = 1

if runPart1:
    if exists:
        
        print('Corresponding clean videoFile already exists or its cleanFrameList exists, nothing to do!!!')
    
            
            
    else:
        print('Corresponding clean videoFile does not exists!!!, this will take a while')
        print('Suggestion: Try to do offline and use GPU for fast processing')
        
        from videoCleaningUsingDNN import videoCleaningWithDNN
        
        modelFile = os.path.join(ROOT_DIR, args_vid.checkpointDir, 'binaryEndoClassifier_124_124.h5')
        frame_scores, nframes, cleanFrameList = videoCleaningWithDNN (modelFile, videoFile, useCleanVideo, PROJECT_DIR, outfile) 
        
        '''
            Identify original video frames that has been cleared for further processing
            This is 0-1 for badframe-good frame
            framescore: score of the frames according to the video quality
            cleanFrameList: appended video frame arrays, reshaped to 124 x 124 x 3
        '''
        np.save(RESULT_DIR+'/'+outfile+'_frameScore', frame_scores)
        np.save(RESULT_DIR+'/'+outfile+'_cleanFrameList', cleanFrameList)
        
        if debug:   
            im = np.reshape(cleanFrameList[0], (124,124,3))
            imageDebugging.showImageDebug(im)
            
        print('video cleaning done... saved frameListBinary and cleanFrameList as noy file, check:', RESULT_DIR)
    
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Encode Video Embedding
    input - clean video
    output - video encoding for clean video
    save as - .npz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
from retrieveQuerySequences import formEmbedding
from imageCroppingClasses import imageCropping

useParser = 0
use3Channel = 1

frameListConcatenated = []
   
# check for existing encoding
embeddingFile= RESULT_DIR +'/'+'AE_'+patientFileName+'_ch_3.npy'
exists = os.path.isfile(embeddingFile)

if exists:
    print('embedding already exists, nothing to do!!!')
    loaded_embedding = 1
    x_train = ''
else:
    # todo: use the video reading function above here!!!!
    print('embedding doesnot exists, please wait while embedding is done!!!')
    print('Loading file 2: ',  RESULT_DIR+'/'+ outfile+'_cleanFrameList'+'.npy')
    cleanFrameList = np.load(RESULT_DIR+'/'+ outfile+'_cleanFrameList'+'.npy')
    frame_scores = np.load(RESULT_DIR+'/'+ outfile+'_frameScore'+'.npy')
    print(len(cleanFrameList))
    
    if (len(frame_scores) == 0):
        print('frame is not clean..')
    else:
        print('Clean videoFrameList loaded...')

    nFramesSelected = [0, 5000]
    val_thresh = 30
    target_shape = (124,124)
    useGray=1
    
    #frameListConcatenated = videoFramesExtraction.extratedVideoFramesInArray(videoFile, nFramesSelected, useGray, target_shape, val_thresh )
    # grayscale image for now::: TODO change to color training
    if use3Channel:
        for i in range (0, len(cleanFrameList)):
            frameListConcatenated.append(np.reshape(cleanFrameList[i], (124,124,3)))
            
    else:
        for i in range (0, len(cleanFrameList)):
            frameListConcatenated.append(imageCropping.read_rgb(np.reshape(cleanFrameList[i], (124,124,1))  ))
        
    print('image files saved in an array, ''todo'' to write in a folder!!!')
    print(' ')
    
    ''' 
        Do embedding here 
    '''
    from keras.models import Model, load_model
    useDebug=0
    print('files being loaded (embedding set to 0 or not available)...\n...this will take a while...')
    
    ''' Start encoding here : these are cleaned frames only  '''
    t1 = time.time()
    x_train = np.reshape(frameListConcatenated, (len(frameListConcatenated), 124, 124, 3) )
    
    print(x_train.shape)
    
    # BE_Autoencoder_124_124_ch3.h5, 33000_AE_1716_11012018-smallCNNFilters.h5
    autoencoder = load_model(os.path.join(ROOT_DIR, args_vid.checkpointDir, 'BE_Autoencoder_124_124_ch3.h5'))
    autoencoder.summary()
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    
    # also saves as embedding file
    learned_codes = formEmbedding(encoder, x_train, embeddingFile, useDebug)
    t2 = time.time()
    
    print('Embedding done in',t2-t1, 'for video with frames: ', len(frameListConcatenated))
    print('summary of my network is:', autoencoder.summary())
    
    print('video cleaning done... saved encoded video asn npy, check:', RESULT_DIR)
    
    
    
