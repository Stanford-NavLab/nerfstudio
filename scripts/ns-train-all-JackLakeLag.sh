#!/bin/bash

# make a directory to save logs
JACKLAKELAGDIRECTORY="nslogging/JackLakeLag"
mkdir -p $JACKLAKELAGDIRECTORY

# create a time a datetime variable for current datetime
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/10p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat10Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag10p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/20p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat20Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag20p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/30p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat30Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag30p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/40p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat40Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag40p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/50p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat50Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag50p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/60p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat60Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag60p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/70p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat70Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag70p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/80p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat80Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag80p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/90p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat90Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag90p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/CombinedVideo/100p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Splat100Percent > ${JACKLAKELAGDIRECTORY}/SplatJackLakeLag100p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/10p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto10Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag10p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/20p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto20Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag20p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/30p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto30Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag30p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/40p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto40Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag40p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/50p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto50Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag50p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/60p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto60Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag60p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/70p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto70Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag70p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/80p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto80Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag80p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/90p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto90Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag90p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train nerfacto --data data/JackLakeLag/CombinedVideo/100p/ --output-dir outputs/JackLakeLag/CombinedVideo --vis wandb --experiment-name Nerfacto100Percent > ${JACKLAKELAGDIRECTORY}/NerfactoJackLakeLag100p_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/20_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat20Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag20Percent_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/30_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat30Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag30Percent_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/40_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat40Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag40Percent_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/50_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat50Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag50Percent_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/60_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat60Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag60Percent_${CURRENTDATETIME}.txt
#CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
#ns-train splatfacto --data data/JackLakeLag/70_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat70Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag70Percent_${CURRENTDATETIME}.txt
CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
ns-train splatfacto --data data/JackLakeLag/80_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat80Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag80Percent_${CURRENTDATETIME}.txt
CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
ns-train splatfacto --data data/JackLakeLag/90_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat90Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag90Percent_${CURRENTDATETIME}.txt
CURRENTDATETIME=`date +"%Y-%m-%d_%H-%M-%S"`
ns-train splatfacto --data data/JackLakeLag/100_percent/ --output-dir outputs/JackLakeLag/ --vis wandb --experiment-name Splat100Percent > ${JACKLAKELAGDIRECTORY}/JackLakeLag100Percent_${CURRENTDATETIME}.txt
