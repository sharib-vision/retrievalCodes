#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:56:39 2018

@author: shariba
retrieveImagesFromEncoding.py
"""

from retrieveQuerySequences import retrieve_closest_images
from imageCroppingClasses import imageProperties
import time
import os
import numpy as np

from paths import DATA_DIR, ROOT_DIR,  RESULT_DIR

# encodings and listImages(124x124) are saved in RESULT_DIR

QueryImageFolder=imageProperties.detect_imgList('images/', '.bmp')
    
'''
Load your compressed files for corresponding video first

'''

import argparse
parser_vid = argparse.ArgumentParser()
parser_vid.add_argument('-videoFile', type=str, default='M_01032018130721_0000000000003059_1_002_001-1', help='enter checkpoint directory')
parser_vid.add_argument('-gpu_id', type=int, default=3, help='enter checkpoint directory')
parser_vid.add_argument('-useTestFlag', type=int, default=1, help='enter checkpoint directory')
args_vid = parser_vid.parse_args()

#TODO parse it
gpu = 1

if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args_vid.gpu_id)
    
if args_vid.useTestFlag:
    args_vid.videoFile='M_01032018130721_0000000000003059_1_002_001-1'
    args_vid.videoFile= 'M_01032017120001_0000000000001603_1_001_001-1'
    args_vid.videoFile= 'M_10082017122840_0000000000002470_1_001_001-1'

args_vid.videoFile=args_vid.videoFile+'.MP4'

print("check me, please: arguments~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(args_vid.videoFile)
print(args_vid.gpu_id)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

videoFileName = args_vid.videoFile
videoFile = os.path.join(DATA_DIR, videoFileName)
patientFileName=videoFile.split('_')[1]+'_'+videoFile.split('_')[2].strip("0")

embeddingFile= RESULT_DIR +'/'+'AE_'+patientFileName+'_ch_3.npy'
print('embedding file is:', embeddingFile)

outfile=videoFileName.split('.')[0]
FrameRetrievalScores=RESULT_DIR+'/'+outfile+'_frameScoreOriginal'
exist_retrievedScores= os.path.isfile(FrameRetrievalScores)
print(FrameRetrievalScores)
print('status:', exist_retrievedScores)

if exist_retrievedScores:
    print('already retrieved scores available')
else:
    exists = os.path.isfile(embeddingFile)
   
    if exists:
        print('embedding already exists, loading using saved imageLists!!!')
        x_train = np.load(RESULT_DIR+'/'+outfile+'_cleanFrameList.npy')
        frame_scores = np.load(RESULT_DIR+'/'+ outfile+'_frameScore'+'.npy')
        print('imageLists of the embedding loaded... continuing to query image file')
        loaded_embedding = 1
    
    
    import cv2
    from skimage.transform import resize
    
    """Prepare your query list
     also try to crop it?
    
    """
    listQueryImages = []
    n_samples=60
    target_shape = (124, 124)
    channel = 3
    useDebug=0
    
    for k in range(0, len(QueryImageFolder)):
        listQueryImages.append(resize(cv2.imread(QueryImageFolder[k]), target_shape))
    
    x_test = np.reshape(listQueryImages, (len(listQueryImages), target_shape[0], target_shape[1], channel) )
    x_train_reshaped=np.reshape(x_train, (len(x_train), target_shape[0], target_shape[1], 3) )
    
    '''Load trained autoencoder here for encoding query samples ===> do parallel or batch for all queries'''
    from keras.models import Model, load_model
    autoencoder = load_model(os.path.join(ROOT_DIR, 'dysplasiaEndoscopy', 'BE_Autoencoder_124_124_ch3.h5'))
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    
    kept_indexes_allQueries = []
    
    print('already learned embedding being loaded ... ')
    learned_codes = np.load(embeddingFile)
    
    # get frames in original video
    frameWithOnes=[i for i,x in enumerate(frame_scores) if x == 1]
    storeOriginalFrames=[]
    
    for k in range(0, len(QueryImageFolder)):
        
        index_test = k
        
        t0 = time.time()
        filenameToSave = RESULT_DIR+'/'+'query_'+str(index_test)+'_AE_'+ patientFileName
        kept_indexes = retrieve_closest_images(x_test[index_test], [] , n_samples, x_train, [], filenameToSave, [], loaded_embedding, learned_codes, [],  index_test, 0, encoder, 0 )
        t1 = time.time()
        print('Retrived image in: ', t1-t0)
        kept_indexes_allQueries.append(kept_indexes)
        # determine the frame numbers in original videos
     
    printQueryImages=0   
    # print query list of images 
    if  printQueryImages:
        retrieved_images = x_test[0]
        for k in range(1, len(QueryImageFolder)):
            retrieved_images = np.hstack((retrieved_images, x_test[k]))
        
        cv2.imwrite('QUERYLIST.jpg',retrieved_images*255)   
     
    '''  Display the corresponding kept_indexes from the saved npy file'''
    display = 0
    keepOnlyClean=0
    
    for l in range (0, len(QueryImageFolder)):
        retrieved_images = []
        retrieveList = []
        orig_frame_score = []
        retrieveFinal = []
        retrieved_images = x_test[l]
        retrieved_images = np.hstack((retrieved_images, x_train_reshaped[kept_indexes_allQueries[l][0]][:,:,[2,1,0]]))
        '''store the retrived image for only frames with ones (i.e. clean frames only) ::::::::::'''
        if keepOnlyClean:
            orig_frame_score.append(frameWithOnes[kept_indexes_allQueries[l][0]])
        else:
            orig_frame_score.append(kept_indexes_allQueries[l][0])
        
        for i in range (1, n_samples-1):
            if i%10==0 and i >0:
                retrieveList.append(retrieved_images)
                retrieved_images=[]
                retrieved_images = x_test[l]
                #retrieved_images = np.hstack((retrieved_images, x_train_reshaped[kept_indexes_allQueries[l][i]][:,:,[2,1,0]]))
            if keepOnlyClean:
                orig_frame_score.append(frameWithOnes[kept_indexes_allQueries[l][i]])
            else:
                orig_frame_score.append(kept_indexes_allQueries[l][i])
                
            retrieved_images = np.hstack((retrieved_images, x_train_reshaped[kept_indexes_allQueries[l][i]][:,:,[2,1,0]]))
            
            if useDebug:
                print((retrieveList[0].shape))
                print((retrieveList[1].shape))
                print(len(retrieveList))
                
        retrieveFinal = retrieveList[0]
        for i in range (0, len(retrieveList)-1):
            retrieveFinal = np.vstack((retrieveFinal, retrieveList[i+1]))
        
        
        if display: 
            from imageCroppingClasses import imageDebugging
            imageDebugging.showImageDebug_matchChannels(retrieved_images)
        
        # 
        fileName = QueryImageFolder[l].split('.')[0]
        #cv2.imwrite(fileName+patientFileName+'.jpg',retrieved_images*255)
        cv2.imwrite(fileName+patientFileName+'.jpg',retrieveFinal*255)
        
        # store the original indexes
        storeOriginalFrames.append(orig_frame_score)
    
    # save original frames retrieved indexes in not clean video
    np.save(RESULT_DIR+'/'+outfile+'_frameScoreOriginal', storeOriginalFrames)   
    
    # orig_frames=np.load(RESULT_DIR+'/'+outfile+'_frameScoreOriginal.npy')    
    '''CheckList'''
    print('EmbeddingFile was: ', )
         
         
    '''TO DO: find the corresponding image in the original video'''
    
    # test and validation files are same
    #retrieve_closest_images(test_element, test_label, n_samples, x_train, y_train, filenameToSave, fileListVideoSeq, 
    #                            loaded_embedding, embeddingFile, validation_files, index_test, useDebug, encoder, cpu):
    
    
    #[i for i,x in enumerate(testlist) if x == 1]
    
    
    
