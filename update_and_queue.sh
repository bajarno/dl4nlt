#!/bin/sh

cd ~

if [ ! -d "dl4nlt" ]; then
  git clone git@github.com:bajarno/dl4nlt.git
  echo 'Cloned git'
fi

cd dl4nlt/code

git fetch origin
git reset --hard origin/master
echo 'Reset to origin/master'

cd ~/dl4nlt
qsub -q run_exp.sh
echo 'Registered job in queue'
