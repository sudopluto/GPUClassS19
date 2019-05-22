#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=exec.%j.out

cd /scratch/$USER/GPUClassS19/HOL5/brightness/

set -o xtrace
./brightness ../input/fractal.pgm 100
#./brightness ../input/world.pgm 100

