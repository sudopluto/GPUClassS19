#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=vAdd
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=vAdd.%j.out

cd /scratch/`whoami`/GPUClassS19/HOL2/

set -o xtrace
./vAdd 100000000 32
echo
./vAdd 100000000 64
echo
./vAdd 100000000 128
echo
./vAdd 100000000 256
echo
./vAdd 100000000 512
