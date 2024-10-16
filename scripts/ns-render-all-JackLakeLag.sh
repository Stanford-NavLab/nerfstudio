#!/bin/bash

CAMERAPATH="/home/navlab/NeRF/nerfstudio/data/JackLakeLag/CombinedVideo/Presentation-Render-2024-08-27_14-49.json"
OUTPUTPATHPREAMBLE="renders/DownsampledFrames/Splatfacto-"
EXTRAPARAMS="--rendered-output-names rgb depth --depth-near-plane 0.001 --depth-far-plane 3.0 --colormap-options.colormap viridis"
NSRENDERPREAMBLE='outputs/JackLakeLag/CombinedVideo'

#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat10Percent/splatfacto/2024-08-26_215155/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}10p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat20Percent/splatfacto/2024-08-26_220300/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}20p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat30Percent/splatfacto/2024-08-26_221406/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}30p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat40Percent/splatfacto/2024-08-26_222502/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}40p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat50Percent/splatfacto/2024-08-26_223600/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}50p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat60Percent/splatfacto/2024-08-26_224659/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}60p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
# CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
# ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat70Percent/splatfacto/2024-08-26_225751/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}70p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}

NSRENDERPREAMBLE='outputs/JackLakeLag'
echo $NSRENDERPREAMBLE
echo ${NSRENDERPREAMBLE}/Splat80Percent/splatfacto/2024-08-27_121525/config.yml


CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat80Percent/splatfacto/2024-08-27_121525/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}80p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat90Percent/splatfacto/2024-08-27_122617/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}90p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
ns-render camera-path --load-config ${NSRENDERPREAMBLE}/Splat100Percent/splatfacto/2024-08-27_123706/config.yml --camera-path-filename ${CAMERAPATH} --output-path ${OUTPUTPATHPREAMBLE}100p-${CURRENTDATETIME}.mp4 ${EXTRAPARAMS}
