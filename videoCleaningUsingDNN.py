#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:08:50 2018

@author: shariba
"""
import numpy as np
    
def read_rgb(f):

    import cv2

    im = cv2.imread(f)
    [b,g,r] = cv2.split(im)

    return cv2.merge([r,g,b])

def detect_imgs(infolder, ext='.tif'):

    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def classify_information(img, model, shape=(64,64)):

    from skimage.transform import resize

    im = resize(img, shape)

    info = model.predict(im[None,:])

    return info


def videoCleaningWithDNN (modelFile, videoFile, useCleanVideoFlag, PROJECT_DIR, outfile):
    
    from keras.models import load_model
    from skimage.transform import resize
    from imageCroppingClasses import videoFramesExtraction, imageCropping

    # todo train with 124, 124
    target_shape = (124, 124)
    
    CNN_classify_model = load_model(modelFile)
    
    clip, (clip_fps, clip_duration, n_frames) = videoFramesExtraction.movieClipinfo(videoFile)    
    boundingbox = videoFramesExtraction.returnBoundingBox(videoFile, 5) 
    print('frame fps:', clip_fps)
    print('bounding box size', boundingbox.shape)
    
    batch_size = 1
    # if batch is different than 1 this will not add up to the n_frames
    frame_scores = []
    cleanFrameList = []
    n_batches = int(np.ceil(n_frames/float(batch_size)))

    from scipy.misc import imsave
    import os

    resultFolder=os.path.join(PROJECT_DIR, outfile)
    os.makedirs(resultFolder, exist_ok=True)
#    clip_fps = 25
    print('saving files:', resultFolder)
    
    for i in range(n_batches)[:]:
        if i <n_batches:
            start = i*batch_size
            end = (i+1)*batch_size
            vid_frames = np.array([resize(imageCropping.crop_image( clip.get_frame(ii*1./clip_fps), boundingbox ), target_shape) for ii in range(start,end,1)]) 
            vid_frames_origShape  = np.array([resize(imageCropping.crop_image( clip.get_frame(ii*1./clip_fps), boundingbox ), (512,512)) for ii in range(start,end,1)])
        else:
            start = i*batch_size
            vid_frames = np.array([resize(imageCropping.crop_image( clip.get_frame(ii*1./clip_fps), boundingbox ), target_shape) for ii in range(start,n_frames,1)])       
            vid_frames_origShape = np.array([resize(imageCropping.crop_image( clip.get_frame(ii*1./clip_fps), boundingbox ), (512,512)) for ii in range(start,n_frames,1)])  

#        if useCleanVideoFlag==1:
        informativeness = CNN_classify_model.predict(vid_frames)
        information_index = np.argmax(informativeness, axis=1)
        frame_scores.append(information_index)
        
        if information_index == 1:
            '''
            Note: extacted frames will directly correspond to frame_score ones 
            [indx] as these numbers are stored only with index values that are positive) 
            frame_scores = np.load(RESULT_DIR+'/' + args_vid.videoFile+'_frameScore.npy')
            frameWithOnes=[i for i,x in enumerate(frame_scores) if x == 1]
            frameRetrievedOriginal = np.load(RESULT_DIR+'/' + args_vid.videoFile+'_frameScoreOriginal.npy')
            indx = frameRetrievedOriginal[14][0]
            correspondingFrame = frameWithOnes[indx]
            '''
            cleanFrameList.append(vid_frames)
            print(resultFolder+'/'+str(i)+'.jpg')
            imsave(resultFolder+'/'+str(i)+'.jpg', np.reshape(vid_frames_origShape, (512,512,3)))
                
#        else: 
#             cleanFrameList.append(vid_frames)
#             information_index = []
#             frame_scores.append(information_index)
             
    frame_scores = np.hstack(frame_scores)
    
    return frame_scores, n_frames, cleanFrameList


def videoCleaningWithDNN_frames (modelFile, imgFolder, patientID, RESULT_DIR):
    
    from keras.models import load_model
#    from shutil import copyfile, move
    
    gpu = 1
    
    if gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=str(3)
    
    CNN_classify_model = load_model(modelFile)
    target_shape = (124, 124)

    imgfiles = detect_imgs(imgFolder, ext='.jpg')
    
    textfilename = patientID+'.txt'

    print(textfilename)
    
    file = RESULT_DIR +'/'+ textfilename
    textfile = open(file, 'a')

    tiles_score = []
    tiles_name = []

    for imagePath in imgfiles[:]:
        #image = io.imread(imagePath)
        print(imagePath)
        # img1 = cv2.imread(imagePath, 1)
        img1 = read_rgb(imagePath)
        informativeness = classify_information(img1, CNN_classify_model, shape=target_shape)
        info_index = np.argmax(informativeness.ravel())
        # >0.51
        tiles_score.append(info_index)

        if info_index==1:
            
            resultFileName=imagePath.split('/')[-1]
            textfile.write('\n'+ resultFileName)
            
            tiles_name.append(resultFileName)
#            if arg_input.moveImages:
#                move(imgFolder+resultFileName, RESULT_DIR+resultFileName)
#            else:
#                copyfile(imgFolder+resultFileName, RESULT_DIR+resultFileName)

    textfile.write('\n')
    textfile.write('\n Total: '+ str(len(tiles_name)))
    textfile.close()
