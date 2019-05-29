#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof_heq
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=nvprof_heq.%j.out

module load opencv/3.4.3-contrib

cd /scratch/$USER/GPUClass19/FINPROJ/heq/

set -o xtrace
nvprof ./heq input/bridge.png
echo "METRICS"
nvprof -m all ./heq input/bridge.png

