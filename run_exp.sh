#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

cd ~
log=$(date +"%Y-%m-%d_%H:%M:%S")
touch $log

echo 'Starting run' >> $log

pip install torch torchvision --user --no-cache -U

echo 'Installed torch' >> $log

if [ ! -d "dl4nlt" ]; then
  git clone git@github.com:bajarno/dl4nlt.git
  echo 'Cloned git' >> $log
fi

git fetch origin
git reset --hard origin/master
echo 'Reset to origin/master' >> $log

cd dl4nlt/code
echo 'Start running python' >> $log

python train.py 2>&1 | tee -a $log

echo 'Done running python' >> $log
