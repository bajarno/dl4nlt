#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

cd ~/dl4nlt/code
logdir="$TMPDIR"/$(date +"%Y-%m-%d_%H:%M:%S")/
mkdir $logdir
touch $logdir/log


echo 'Starting run' | tee -a "$logdir"/log
pip install torch torchvision --user --no-cache -U
echo 'Installed torch' | tee -a "$logdir"/log

echo 'Start running python' | tee -a "$logdir"/log
python -u train.py --output_dir $logdir 2>&1 | tee -a "$logdir"/log
echo 'Done running python' | tee -a "$logdir"/log

cp -r $logdir ~
