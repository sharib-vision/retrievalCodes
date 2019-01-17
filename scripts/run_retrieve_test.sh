#!/bin/bash

CURRENT_DIR=`pwd`

checkpointDir=checkpoints
vidFile=712001

gpuNo=3

echo " ===== encoding videos, chill it will take some time if it doesnot exist=========="
${CURRENT_DIR}/encodeVideos.py -videoFile $vidFile -gpu_id $gpuNo -useTestFlag 0 -checkpointDir $checkpointDir



