#!/bin/bash

gpu=1
if [ $gpu == "1" ]
then
    source activate TFPytorchGPU
    module load cuda/9.0
else
    echo "CPU option taken"
    source activate kerasTFCPU
    module load cuda/9.0
fi

USER=`whoami`

echo $USER

if [ $USER == "shariba" ]
then
    ROOT_FOLDER=/Volumes/rescomp2/data
else
    ROOT_FOLDER=/well/rittscher/users/sharib
fi


####################################################################################
### Choose videos
####################################################################################
declare -a videoArray
videoArray=( 'M_04012018112856_0000000000002042_1_001_001-1')

echo "========Default settings into place...==================="
startTime=`date +%s`
gpuNumber=3
RetrievalDir=dysplasiaRetrievedList
echo " ===== encoding videos, chill it will take some time if it doesnot exist=========="
CODE_FOLDER=$ROOT_FOLDER/dysplasiaEndoscopy
${CODE_FOLDER}/encodeVideos.py -videoFile ${videoArray[0]} -gpu_id $gpuNumber -useTestFlag 0
echo " ===== retrieval, should be fairly fast=========="
${CODE_FOLDER}/retrieveImagesFromEncoding.py -videoFile ${videoArray[0]} -gpu_id $gpuNumber -useTestFlag 0
echo " ===== retrieving to a list.txt...=========="
${CODE_FOLDER}/retrieval_direct.py -videoFile ${videoArray[0]} -useTestFlag 0
echo " ===== copying files...=========="
${CODE_FOLDER}/copyImages.py --txtFile ${videoArray[0]}'_retrievedImages.txt' --retrivalDir $RetrievalDir

runTime=$(expr `date +%s` - $startTime)

echo "====================================================================================================================="

echo "===   Done with execution!                                          ================================================="
echo "===   Current time: `date`                                          ================================================="
echo "===   Run time: $(expr $runTime / 1    )s  ( ~= $(expr $runTime / 60   )m ~= $(expr $runTime / 3600 )h)                    ================================================="
echo "====================================================================================================================="

echo "Finished at :"`date`
exit 0


