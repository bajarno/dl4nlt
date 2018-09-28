#!/bin/sh

cd ~

if [ ! -d "dl4nlt" ]; then
  git clone git@github.com:bajarno/dl4nlt.git
  echo 'Cloned git' | tee -a $log
fi

cd dl4nlt/code

git fetch origin
git reset --hard origin/master
echo 'Reset to origin/master' | tee -a $log

cd ~/dl4nlt
qsub -q run_exp.sh
